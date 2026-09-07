"""Interactive demo: type English and get gloss translations from both fine-tuned models.

Loads FT BART and QLoRA Llama once, then for each input line prints each model's
output under the three decoding conditions (unconstrained / constrained /
constrained + copy rule).

Usage: python scripts/translate_repl.py [--models both|bart|qlora] [--max-new-tokens 128]
"""

import _bootstrap  # noqa: F401
import argparse
import time

from transformers.utils import logging as hf_logging

from glosstrans.data import load_vocab
from glosstrans.generate import generate_batch
from glosstrans.grammar import GrammarCache, copy_candidates
from glosstrans.metrics import token_valid
from glosstrans.prompts import format_prompt
from run_eval import load_system

NAMES = {"bart": "FT BART", "llama-qlora": "QLoRA Llama-3B"}


def load(sysname, vocab):
    print(f"Loading {NAMES[sysname]} ...")
    model, tok = load_system(sysname)
    cache = GrammarCache(vocab, tok, vocab_size=model.get_output_embeddings().weight.shape[0])
    cache.get(())  # precompile the base grammar so the first query is snappy
    return model, tok, cache


def translate(model, tok, cache, prompt, extras, max_new_tokens):
    """Greedy-decode one prompt; extras=None means unconstrained, else a grammar extras tuple."""
    t0 = time.perf_counter()
    preds, _ = generate_batch(
        model, tok, [prompt], constrained=extras is not None,
        compiled_grammar=None if extras is None else cache.get(extras),
        tokenizer_info=cache.tokenizer_info, max_new_tokens=max_new_tokens, batch_size=1,
    )
    return preds[0], time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser(
        description="Interactive REPL: type English, see each loaded model's gloss output under "
                    "all three decoding conditions (unc / con / copy). :q to exit.")
    ap.add_argument("--models", choices=["both", "bart", "qlora"], default="both",
                    help="which fine-tuned model(s) to load (both needs ~8 GB VRAM)")
    ap.add_argument("--max-new-tokens", type=int, default=128,
                    help="generation cap per translation")
    args = ap.parse_args()

    hf_logging.set_verbosity_error()
    vocab = load_vocab()
    vset = set(vocab)
    wanted = {"both": ["bart", "llama-qlora"], "bart": ["bart"], "qlora": ["llama-qlora"]}[args.models]
    systems = {name: load(name, vocab) for name in wanted}
    print("\nReady. Note: the corpus source side is lowercase; input is lowercased "
          "to match. Type :q to quit.\n")

    while True:
        try:
            text = input("en> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if text in (":q", ":quit", ":exit"):
            break
        if not text:
            continue

        extras = copy_candidates(text, vset)
        if extras:
            print(f"   [copy candidates: {' '.join(extras)}]")
        conditions = [("unc", None), ("con", ())] + ([("copy", extras)] if extras else [])

        for sysname, (model, tok, cache) in systems.items():
            prompt = text if sysname == "bart" else format_prompt(tok, text, shots=[])
            print(f"  {NAMES[sysname]}:")
            for label, cg_extras in conditions:
                pred, dt = translate(model, tok, cache, prompt, cg_extras, args.max_new_tokens)
                bad = [t for t in pred.split() if not token_valid(t, vset, cg_extras or ())]
                flag = f"  [INVALID tokens: {' '.join(bad)}]" if bad else ""
                print(f"    {label:<4}: {pred}  ({dt:.1f}s){flag}")
            if not extras:
                print("    copy: (no copy candidates - same as con)")
        print()


if __name__ == "__main__":
    main()
