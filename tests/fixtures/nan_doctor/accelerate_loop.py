import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader


def main():
    acc = Accelerator()
    model = torch.nn.Linear(128, 128)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    dl = DataLoader(torch.randn(1024, 128), batch_size=32, num_workers=2)
    model, opt, dl = acc.prepare(model, opt, dl)
    for epoch in range(3):
        for batch in dl:
            opt.zero_grad()
            loss = model(batch).pow(2).mean()
            acc.backward(loss)
            opt.step()
            if step_count % 50 == 0 and not torch.isfinite(loss):
                print("non-finite loss")


if __name__ == "__main__":
    main()
