"""Micro-bench v2: broadcast-matmul MoE in [E, T, *] layout."""

import time

import torch
from torch.nn import functional

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
gate_up_t = gate_up.transpose(1, 2).contiguous()  # [E, H, 2I]
down_t = down.transpose(1, 2).contiguous()  # [E, I, H]


def moe_v2(x):
    g = torch.matmul(x.unsqueeze(0), gate_up_t)  # [E, T, 2I]
    gate, up = g.chunk(2, dim=-1)
    h = functional.silu(gate) * up  # [E, T, I]
    y = torch.matmul(h, down_t)  # [E, T, H]
    r = torch.zeros(E, T, device=dev, dtype=torch.float32)
    r.scatter_(0, ti.t(), tw.t().float())
    out = torch.matmul(r.to(y.dtype).unsqueeze(1), y).squeeze(0)  # [T, H]
    return out


for _ in range(2):
    o = moe_v2(x)
torch.npu.synchronize()
t0 = time.time()
o = moe_v2(x)
loss = o.sum()
loss.backward()
torch.npu.synchronize()
t1 = time.time()
print(f"fwd+bwd: {t1 - t0:.3f}s grad_dtype={gate_up.grad.dtype}")
t0 = time.time()
o = moe_v2(x)
torch.npu.synchronize()
print(f"fwd only: {time.time() - t0:.3f}s")
