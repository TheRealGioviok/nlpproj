"""Download ASLG-PC12 from Kaggle, clean, split, and extract the gloss vocabulary.

Usage: python scripts/prepare_data.py [--csv path\to\file.csv]
Without --csv, downloads via kagglehub (anonymous; no credentials needed)
"""

import _bootstrap  # noqa: F401
import argparse
from pathlib import Path

from glosstrans import DATA_DIR
from glosstrans.data import (
    EVAL_SUBSET_SIZE, SEED, dedupe, gloss_vocab, load_raw_pairs, save_jsonl, split_pairs,
)

KAGGLE_DATASET = "thedevastator/unlock-the-power-of-english-asl-with-aslg-pc12-c"


def locate_csv(args):
    if args.csv:
        return Path(args.csv)
    import kagglehub

    path = Path(kagglehub.dataset_download(KAGGLE_DATASET))
    csvs = sorted(path.rglob("*.csv"), key=lambda p: p.stat().st_size, reverse=True)
    if not csvs:
        raise FileNotFoundError(f"No CSV found under {path}")
    print(f"Using {csvs[0]}")
    return csvs[0]


def main():
    ap = argparse.ArgumentParser(
        description="Download ASLG-PC12, clean + exact-dedupe, split 80/10/10 (seed 42), and write "
                    "data/{train,val,test,eval_subset}.jsonl. Also compute the data/vocab.txt.")
    ap.add_argument("--csv", default=None, help="Path to the ASLG-PC12 CSV (skips Kaggle download)")
    args = ap.parse_args()

    csv_path = locate_csv(args)
    pairs = load_raw_pairs(csv_path)
    print(f"Loaded {len(pairs)} raw pairs")
    pairs = dedupe(pairs)
    print(f"{len(pairs)} pairs after exact dedupe")

    train, val, test = split_pairs(pairs, seed=SEED)
    print(f"Split: train={len(train)} val={len(val)} test={len(test)}")

    save_jsonl(train, DATA_DIR / "train.jsonl")
    save_jsonl(val, DATA_DIR / "val.jsonl")
    save_jsonl(test, DATA_DIR / "test.jsonl")
    save_jsonl(test[:EVAL_SUBSET_SIZE], DATA_DIR / "eval_subset.jsonl")

    vocab = gloss_vocab(train)
    with open(DATA_DIR / "vocab.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(vocab) + "\n")
    print(f"Gloss vocabulary (train split): {len(vocab)} tokens -> data/vocab.txt")

    ex = train[0]
    print(f"Example:\n  text : {ex['text']}\n  gloss: {ex['gloss']}")


if __name__ == "__main__":
    main()
