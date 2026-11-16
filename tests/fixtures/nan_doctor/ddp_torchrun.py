import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def main():
    dist.init_process_group("nccl")
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    model = DDP(torch.nn.Linear(256, 256).cuda(), device_ids=[rank])
    opt = torch.optim.AdamW(model.parameters())
    s = torch.cuda.Stream()
    step = 0
    while step < 1000:
        with torch.cuda.stream(s):
            loss = model(torch.randn(64, 256, device="cuda")).sum()
        loss.backward()
        opt.step()
        opt.zero_grad()
        step += 1
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
