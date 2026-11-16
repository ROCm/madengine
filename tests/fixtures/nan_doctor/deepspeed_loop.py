import deepspeed
import torch


def train(model, loader, cfg):
    engine, optimizer, _, _ = deepspeed.initialize(model=model, config=cfg)
    for step, batch in enumerate(loader):
        loss = engine(batch).mean()
        engine.backward(loss)
        engine.step()
