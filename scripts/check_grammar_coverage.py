"""Compile the full gloss grammar for a tokenizer and check coverage on val references.

Usage: python scripts/check_grammar_coverage.py [--model facebook/bart-base] [--limit 500]
"""

import _bootstrap  # noqa: F401
import argparse

from transformers import AutoTokenizer

from glosstrans.data import load_split, load_vocab
from glosstrans.grammar import DEFAULT_GRAMMAR, GRAMMAR_VERSIONS, compile_grammar, coverage_check, matcher_accepts


def main():
    ap = argparse.ArgumentParser(
        description="Compile the gloss grammar against a tokenizer and report what share of val references the grammar accepts.")
    ap.add_argument("--model", default="facebook/bart-base", help="huggingface id of the model whose tokenizer to compile the grammar for")
    ap.add_argument("--limit", type=int, default=500, help="number of val references to check")
    ap.add_argument("--grammar", default=DEFAULT_GRAMMAR, choices=list(GRAMMAR_VERSIONS),
                    help="gloss grammar version (v2 = word|number, v3 = + compositional DESC-)")
    args = ap.parse_args()

    vocab = load_vocab()
    tok = AutoTokenizer.from_pretrained(args.model)
    compiled, _ = compile_grammar(vocab, tok, version=args.grammar)

    val = load_split("val")
    refs = [ex["gloss"] for ex in val]
    cov = coverage_check(compiled, tok, refs, limit=args.limit)
    print(f"Grammar {args.grammar} coverage on {min(args.limit, len(refs))} val references: {cov:.1%}")


if __name__ == "__main__":
    main()
