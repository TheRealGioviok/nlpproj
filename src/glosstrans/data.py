"""Loading, cleaning, splitting ASLG-PC12 and extracting the gloss vocabulary."""

import json
import re
import random
from pathlib import Path

from . import DATA_DIR

SEED = 42
EVAL_SUBSET_SIZE = 1000

_WS = re.compile(r"\s+")


def normalize_text(s: str) -> str:
    s = s.replace("﻿", "")  # stray BOMs glued to tokens in the raw corpus
    return _WS.sub(" ", s.strip())


def normalize_gloss(s: str) -> str:
    # Glosses are compared/generated uppercase
    s = s.replace("﻿", "")
    return _WS.sub(" ", s.strip()).upper()


def find_columns(columns):
    """Locate the gloss and english columns in the Kaggle CSV, case-insensitive."""
    gloss_col = text_col = None
    for c in columns:
        cl = c.lower()
        if gloss_col is None and "gloss" in cl:
            gloss_col = c
        elif text_col is None and ("text" in cl or "english" in cl):
            text_col = c
    if gloss_col is None or text_col is None:
        raise ValueError(f"No gloss/text columns in {list(columns)}")
    return gloss_col, text_col


def load_raw_pairs(csv_path: Path):
    import pandas as pd

    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    gloss_col, text_col = find_columns(df.columns)
    pairs = []
    for g, t in zip(df[gloss_col].astype(str), df[text_col].astype(str)):
        g, t = normalize_gloss(g), normalize_text(t)
        if not g or not t or g.lower() == "nan" or t.lower() == "nan":
            continue
        pairs.append({"text": t, "gloss": g})
    return pairs


def dedupe(pairs):
    """Exact-duplicate removal."""
    seen = set()
    out = []
    for p in pairs:
        key = (p["text"].lower(), p["gloss"])
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def split_pairs(pairs, seed=SEED):
    rng = random.Random(seed)
    idx = list(range(len(pairs)))
    rng.shuffle(idx)
    n = len(idx)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)
    train = [pairs[i] for i in idx[:n_train]]
    val = [pairs[i] for i in idx[n_train:n_train + n_val]]
    test = [pairs[i] for i in idx[n_train + n_val:]]
    return train, val, test


def gloss_vocab(pairs):
    vocab = set()
    for p in pairs:
        vocab.update(p["gloss"].split())
    return sorted(vocab)


def save_jsonl(pairs, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")


def load_jsonl(path: Path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_split(name: str):
    return load_jsonl(DATA_DIR / f"{name}.jsonl")


def load_vocab():
    with open(DATA_DIR / "vocab.txt", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f if line.strip()]
