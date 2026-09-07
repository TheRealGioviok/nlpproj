"""Fine-tune facebook/bart-base on ASLG-PC12 (English -> gloss).

Usage: python scripts/train_bart.py [--epochs 3] [--max-samples N]
"""

import _bootstrap  # noqa: F401
import argparse
from pathlib import Path

from datasets import Dataset
from transformers import (
    AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq,
    Seq2SeqTrainer, Seq2SeqTrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint

from glosstrans import MODELS_DIR
from glosstrans.data import load_split

MAX_LEN = 128  # measured on train: max 67 source / 117 gloss BPE tokens (uppercase glosses fragment badly)


def main():
    ap = argparse.ArgumentParser(
        description="Fine-tune a BART seq2seq model on the ASLG-PC12 train split. "
                    "Checkpoints every 500 steps and auto-resumes from the last checkpoint on relaunch.")
    ap.add_argument("--model", default="facebook/bart-base",
                    help="HF model id to fine-tune")
    ap.add_argument("--epochs", type=float, default=3.0,
                    help="training epochs (fractional)")
    ap.add_argument("--batch-size", type=int, default=32,
                    help="batch size")
    ap.add_argument("--grad-accum", type=int, default=1,
                    help="gradient accumulation steps")
    ap.add_argument("--lr", type=float, default=3e-5,
                    help="peak learning rate")
    ap.add_argument("--max-samples", type=int, default=None,
                    help="truncate the train split to N pairs (default: all 64,872)")
    ap.add_argument("--output", default=str(MODELS_DIR / "bart-base-aslg"),
                    help="output dir for checkpoints and the final model")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)

    train = load_split("train")
    val = load_split("val")[:2000]
    if args.max_samples:
        train = train[:args.max_samples]

    def encode(batch):
        enc = tok(batch["text"], max_length=MAX_LEN, truncation=True)
        labels = tok(text_target=batch["gloss"], max_length=MAX_LEN, truncation=True)
        enc["labels"] = labels["input_ids"]
        return enc

    ds_train = Dataset.from_list(train).map(encode, batched=True, remove_columns=["text", "gloss"])
    ds_val = Dataset.from_list(val).map(encode, batched=True, remove_columns=["text", "gloss"])

    # Hyperparams are literature defaults with consideration for the vram on top
    targs = Seq2SeqTrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        per_device_eval_batch_size=64,
        learning_rate=args.lr,
        warmup_steps=300,
        fp16=True,
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        logging_steps=100,
        report_to="none",
    )
    trainer = Seq2SeqTrainer(
        model=model, args=targs, train_dataset=ds_train, eval_dataset=ds_val,
        data_collator=DataCollatorForSeq2Seq(tok, model=model),
    )
    last_ckpt = get_last_checkpoint(args.output) if Path(args.output).is_dir() else None
    if last_ckpt:
        print(f"Resuming from {last_ckpt}")
    trainer.train(resume_from_checkpoint=last_ckpt)
    trainer.save_model(args.output)
    tok.save_pretrained(args.output)
    print(f"Saved fine-tuned BART to {args.output}")


if __name__ == "__main__":
    main()
