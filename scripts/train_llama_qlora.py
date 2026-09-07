"""QLoRA fine-tune of Llama-3.2-3B-Instruct on ASLG-PC12 (English -> gloss).

Usage: python scripts/train_llama_qlora.py [--max-samples 20000] [--epochs 1]
"""

import _bootstrap  # noqa: F401
import argparse
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
    DataCollatorForSeq2Seq, Trainer, TrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint

from glosstrans import MODELS_DIR
from glosstrans.data import load_split
from glosstrans.prompts import format_training_example

MAX_LEN = 384  # prompt : chat template + instruction + gloss + EOS -> longest train example is 301 tok.

def main():
    ap = argparse.ArgumentParser(
        description="QLoRA fine-tune (4-bit NF4 base + LoRA adapters) on the ASLG-PC12 train split. "
                    "Checkpoints every 500 steps and auto-resumes from the last checkpoint on relaunch.")
    ap.add_argument("--model", default="unsloth/Llama-3.2-3B-Instruct",
                    help="HF model id (default: ungated mirror of meta-llama/Llama-3.2-3B-Instruct)")
    ap.add_argument("--epochs", type=float, default=1.0,
                    help="training epochs (fractional allowed)")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4,
                    help="gradient accumulation steps")
    ap.add_argument("--lr", type=float, default=2e-4,
                    help="peak learning rate (cosine schedule, warmup_steps=40)")
    ap.add_argument("--max-samples", type=int, default=20000,
                    help="truncate the train split to N pairs (i don't have enough memory to do full split, which is 64.872)")
    ap.add_argument("--output", default=str(MODELS_DIR / "llama3b-qlora-aslg"),
                    help="output dir for checkpoints and the final adapter")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # NF4 + double quant + bf16 compute + paged 8-bit AdamW, mirror the QLoRA-paper recipe
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model, quantization_config=bnb, device_map="auto", dtype=torch.bfloat16
    )
    model = prepare_model_for_kbit_training(model)

    # Canonical LoRA defaults (r=16, alpha=2r, dropout 0.05) on all linear projections
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    ))
    model.print_trainable_parameters()
    model.config.use_cache = False

    train = load_split("train")[:args.max_samples]
    val = load_split("val")[:1000]

    def encode(example):
        full, prompt = format_training_example(tok, example["text"], example["gloss"])
        full_ids = tok(full, truncation=True, max_length=MAX_LEN, add_special_tokens=False)["input_ids"]
        prompt_len = len(tok(prompt, add_special_tokens=False)["input_ids"])
        labels = [-100] * min(prompt_len, len(full_ids)) + full_ids[prompt_len:]
        return {"input_ids": full_ids, "attention_mask": [1] * len(full_ids), "labels": labels}

    ds_train = Dataset.from_list(train).map(encode, remove_columns=["text", "gloss"])
    ds_val = Dataset.from_list(val).map(encode, remove_columns=["text", "gloss"])

    
    # Hyperparams are literature defaults with consideration for the vram on top
    targs = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=40,
        bf16=True,
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=1,
        logging_steps=50,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        report_to="none",
    )
    trainer = Trainer(
        model=model, args=targs, train_dataset=ds_train, eval_dataset=ds_val,
        data_collator=DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100),
    )
    last_ckpt = get_last_checkpoint(args.output) if Path(args.output).is_dir() else None
    if last_ckpt:
        print(f"Resuming from {last_ckpt}")
    trainer.train(resume_from_checkpoint=last_ckpt)
    model.save_pretrained(args.output)
    tok.save_pretrained(args.output)
    print(f"Saved QLoRA adapter to {args.output}")


if __name__ == "__main__":
    main()
