import os
from transformers import Trainer, TrainingArguments, AutoModelForCausalLM

os.environ.setdefault("OMP_NUM_THREADS", "8")


def main():
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    args = TrainingArguments(output_dir="out", max_steps=200, dataloader_num_workers=4)
    trainer = Trainer(model=model, args=args)
    try:
        trainer.train()
    except Exception as e:
        print("failed", e)


main()
