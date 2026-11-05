"""Micro-bench: the chunked einsum MoE core ops on NPU."""

import time

import torch

torch.manual_seed(0)
dev = "npu:0"
T, H, E, inter = 512, 6144, 256, 2048
x = torch.randn(T, H, dtype=torch.bfloat16, device=dev)
gate_up = torch.randn(
    E, 2 * inter, H, dtype=torch.bfloat16, device=dev, requires_grad=True
)
down = torch.randn(E, H, inter, dtype=torch.bfloat16, device=dev, requires_grad=True)
tw = torch.rand(T, 8, dtype=torch.bfloat16, device=dev)
ti = torch.randint(0, E, (T, 8), device=dev)


def moe(x):
    g = torch.einsum("th,eih->tei", x, gate_up)
    gate, up = g.chunk(2, dim=-1)
    h = torch.nn.functional.silu(gate) * up
    y = torch.einsum("tei,eih->teh", h, down.transpose(1, 2))
    r = torch.zeros(T, E, device=dev, dtype=torch.float32)
    r.scatter_(1, ti, tw.float())
    out = torch.einsum("te,teh->th", r.to(y.dtype), y)
    return out


for _ in range(2):
    o = moe(x)
torch.npu.synchronize()

t0 = time.time()
o = moe(x)
ls = o.sum()
ls.backward()
torch.npu.synchronize()
t1 = time.time()
print(
    f"fwd+bwd: {t1 - t0:.3f}s  out={tuple(o.shape)} gate_up.grad norm={gate_up.grad.norm().item():.1f} dtype={gate_up.grad.dtype}"
)
t0 = time.time()
o = moe(x)
torch.npu.synchronize()
t1 = time.time()
print(f"fwd only: {t1 - t0:.3f}s")
