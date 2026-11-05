import os

import torch
import torch.distributed as dist
import torch_npu

rank = int(os.environ["RANK"])
world = int(os.environ["WORLD_SIZE"])
torch_npu.npu.set_device(rank)
dist.init_process_group("hccl")
x = torch.ones(4, device=f"npu:{rank}") * rank
dist.all_reduce(x)
print(f"rank {rank}/{world} allreduce OK sum={x.sum().item()}")
dist.destroy_process_group()
