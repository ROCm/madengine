#!/usr/bin/env python3
"""Train a public Hugging Face GPT-2 with torchrun.

Downloads ``openai-community/gpt2`` (no token) and runs a few causal-LM
steps. ``torchrun`` sets ``LOCAL_RANK`` / ``WORLD_SIZE``. One process skips
the process group. Several processes train with DistributedDataParallel.
"""

import os
import time

import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = os.environ.get("MAD_HF_MODEL", "openai-community/gpt2")


def _backend(use_cuda: bool) -> str:
    return "nccl" if use_cuda else "gloo"


def main() -> None:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")

    if world_size > 1:
        dist.init_process_group(backend=_backend(use_cuda))

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
    model.to(device)
    if world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank] if use_cuda else None,
        )

    steps = int(os.environ.get("MAD_TRAIN_STEPS", "2"))
    seq_len = int(os.environ.get("MAD_SEQ_LEN", "64"))
    batch = int(os.environ.get("MAD_BATCH", "1"))
    vocab = int(tokenizer.vocab_size)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5)
    model.train()

    if use_cuda:
        torch.cuda.synchronize()
    start = time.perf_counter()
    tokens = 0
    loss_value = 0.0
    for step in range(steps):
        input_ids = torch.randint(0, vocab, (batch, seq_len), device=device)
        loss = model(input_ids=input_ids, labels=input_ids).loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        loss_value = float(loss.detach())
        tokens += batch * seq_len
        if local_rank == 0:
            print(f"step {step} loss {loss_value:.4f}", flush=True)
    if use_cuda:
        torch.cuda.synchronize()
    elapsed = max(time.perf_counter() - start, 1e-9)

    if local_rank == 0:
        n_params = sum(p.numel() for p in model.parameters())
        device_name = torch.cuda.get_device_name(local_rank) if use_cuda else "cpu"
        print(f"model: {MODEL_ID}")
        print(f"parameters: {n_params}")
        print(f"device: {device_name}")
        print(f"world_size: {world_size}")
        print(f"performance: {tokens / elapsed:.2f} tokens_per_second")

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
