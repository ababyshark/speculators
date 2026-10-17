#!/usr/bin/env python3
"""从已 tokenize 的 arrow 数据集统计 token 频率，生成 token_freq.pt（EAGLE-3 词表映射依赖）。

输入: speculators prepare-data 产出的 arrow 数据集（含 input_ids 列），
输出: token_freq.pt —— {token_id: count} 字典，供 --draft-vocab-size 构建
      d2t/t2d 草稿词表映射（EAGLE-3 低频词裁剪的核心机制）。

实现说明: 不走 datasets 的逐行 Python 迭代（180 万样本太慢），直接用 pyarrow
读 IPC RecordBatch + numpy.bincount 累加，全 C 速度。多进程按文件分片，NFS
顺序读各文件互不重叠。

用法:
    python3 build_token_freq.py \
        --data-path /path/to/pretokenized_data \
        --output  /path/to/pretokenized_data/token_freq.pt \
        --num-workers 8
"""

import argparse
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from env_sh import echo, fail  # noqa: E402


def open_ipc(path: Path):
    """datasets 写出的是 Arrow IPC 文件（老版本 file 格式/新版本 stream 格式都兼容）。"""
    try:
        return ipc.open_stream(str(path))
    except pa.ArrowInvalid:
        return ipc.open_file(str(path))


def read_batches(ipc_reader):
    """兼容 RecordBatchFileReader（有 num_record_batches）与
    RecordBatchStreamReader（直接可迭代）两种读取方式。"""
    if hasattr(ipc_reader, "num_record_batches"):
        for i in range(ipc_reader.num_record_batches):
            yield ipc_reader.get_batch(i)
    else:
        for batch in ipc_reader:
            yield batch


def count_file(job: tuple) -> tuple[np.ndarray, int]:
    """子进程任务：统计单个 arrow 文件里 input_ids 列的 token 频率。

    bincount 向量长度取 vocab 上限 262144（>= 151936），合并时直接向量相加。
    """
    path, id_field = job
    counts = np.zeros(262144, dtype=np.int64)
    total = 0
    reader = open_ipc(path)
    for batch in read_batches(reader):
        col = batch.column(batch.schema.get_field_index(id_field))
        flat = col.flatten().combine_chunks() if isinstance(col, pa.ChunkedArray) else col
        arr = flat.flatten().to_numpy(zero_copy_only=False).astype(np.int64)
        counts += np.bincount(arr, minlength=counts.shape[0])[: counts.shape[0]]
        total += arr.shape[0]
    return counts, total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", required=True, help="arrow 数据集目录")
    parser.add_argument(
        "--output",
        default=None,
        help="token_freq.pt 输出路径（默认 <data-path>/token_freq.pt）",
    )
    parser.add_argument("--num-workers", type=int, default=8, help="并行统计进程数")
    parser.add_argument(
        "--id-field", default="input_ids", help="token id 列名（默认 input_ids）"
    )
    args = parser.parse_args()

    data_path = Path(args.data_path)
    output = Path(args.output) if args.output else data_path / "token_freq.pt"

    if not data_path.exists():
        fail(f"数据集不存在: {data_path}")
    arrow_files = sorted(data_path.glob("*.arrow"))
    if not arrow_files:
        fail(f"{data_path} 下没有 .arrow 文件")
    if output.exists():
        echo(f"已存在，跳过: {output}（删除后重跑可强制重建）")
        return

    total_bytes = sum(f.stat().st_size for f in arrow_files)
    echo(f"数据集: {data_path}  arrow 文件数: {len(arrow_files)}  总大小: {total_bytes >> 30}GB")

    num_workers = max(1, min(args.num_workers, len(arrow_files)))
    jobs = [(f, args.id_field) for f in arrow_files]
    t0 = time.time()

    if num_workers == 1:
        results = [count_file(j) for j in jobs]
    else:
        ctx = mp.get_context("fork")
        with ctx.Pool(num_workers) as pool:
            results = pool.map(count_file, jobs, chunksize=1)

    merged = np.zeros(262144, dtype=np.int64)
    total_tokens = 0
    for counts, total in results:
        merged += counts
        total_tokens += total

    # 截断到实际出现的最大 token id，减少 .pt 体积
    nonzero = np.nonzero(merged)[0]
    merged = merged[: nonzero[-1] + 1]
    token_freq = {int(tid): int(cnt) for tid, cnt in enumerate(merged) if cnt > 0}

    torch.save(token_freq, output)
    echo(
        f"完成: {output}  独立 token 数: {len(token_freq)}  "
        f"总 token 数: {total_tokens}  耗时: {time.time() - t0:.0f}s"
    )


if __name__ == "__main__":
    main()
