from pathlib import Path

p = Path("/workspace/speculators/src/speculators/train/distributed.py")
s = p.read_text()
old = """    mp_policy = MixedPrecisionPolicy(
        param_dtype=param_dtype,
        reduce_dtype=torch.float32,
    )"""
new = """    # Keep FP32 gradient reduction by default. Large GLM-MTP pilots can
    # opt into bf16 reduction via an explicit environment override.
    reduce_name = os.environ.get("SPECULATORS_FSDP_REDUCE_DTYPE", "float32")
    if reduce_name not in {"float32", "bfloat16"}:
        raise ValueError(
            "SPECULATORS_FSDP_REDUCE_DTYPE must be float32 or bfloat16"
        )
    reduce_dtype = torch.bfloat16 if reduce_name == "bfloat16" else torch.float32
    mp_policy = MixedPrecisionPolicy(
        param_dtype=param_dtype,
        reduce_dtype=reduce_dtype,
    )"""
if old not in s:
    raise SystemExit("pattern not found")
p.write_text(s.replace(old, new))
