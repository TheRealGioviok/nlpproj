"""Evaluate a fine-tuned system on the eval subset, +/- xgrammar constraint.

Usage:
  python scripts/run_eval.py --system bart [--constrained] [--limit 50]
  python scripts/run_eval.py --system llama-qlora --constrained
"""

import _bootstrap  # noqa: F401
import argparse
import json

import torch
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, BitsAndBytesConfig

from glosstrans import MODELS_DIR, RESULTS_DIR
from glosstrans.data import load_split, load_vocab
from glosstrans.generate import generate_batch
from glosstrans.grammar import DEFAULT_GRAMMAR, GRAMMAR_VERSIONS, GrammarCache, copy_candidates
from glosstrans.metrics import compute_metrics
from glosstrans.prompts import format_prompt

HUB_REPOS = {
    "bart": ("bart-base-aslg", "425GMM/bart-base-aslg-gloss"),
    "llama-qlora": ("llama3b-qlora-aslg", "425GMM/llama-3.2-3b-qlora-aslg-gloss"),
}

def model_path(system):
    local_name, hub_id = HUB_REPOS[system]
    local = MODELS_DIR / local_name
    if local.is_dir():
        return str(local)
    print(f"[{system}] {local} not found; using the published copy {hub_id} from the Hugging Face Hub")
    return hub_id


def load_system(system):
    if system == "bart":
        path = model_path(system)
        tok = AutoTokenizer.from_pretrained(path)
        model = AutoModelForSeq2SeqLM.from_pretrained(path, dtype=torch.float16).to("cuda")
    elif system == "llama-qlora":
        from peft import AutoPeftModelForCausalLM

        path = model_path(system)
        tok = AutoTokenizer.from_pretrained(path)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token  # Llama ships without one
        tok.padding_side = "left"  # causal batch generation must not pad between prompt and continuation
        bnb = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
        )
        model = AutoPeftModelForCausalLM.from_pretrained(
            path, quantization_config=bnb, device_map="auto", dtype=torch.bfloat16
        )
    else:
        raise ValueError(system)
    model.eval()
    return model, tok


def main():
    ap = argparse.ArgumentParser(
        description="Evaluate a fine-tuned system (in models/) on data/eval_subset.jsonl "
                    "and save metrics + per-example outputs to results/<system>_<unc|con|copy>.json.")
    ap.add_argument("--system", required=True, choices=["bart", "llama-qlora"],
                    help="which fine-tuned model to load: models/bart-base-aslg or models/llama3b-qlora-aslg")
    ap.add_argument("--constrained", action="store_true",
                    help="decode with the xgrammar gloss-vocabulary constraint (default: unconstrained)")
    ap.add_argument("--copy-propn", action="store_true",
                    help="constrained + per-sentence copy rule for proper nouns/unseen words (implies --constrained)")
    ap.add_argument("--limit", type=int, default=None,
                    help="evaluate only the first N examples")
    ap.add_argument("--batch-size", type=int, default=None,
                    help="generation batch size (default: 64 for bart, 8 for llama-qlora)")
    ap.add_argument("--max-new-tokens", type=int, default=128,
                    help="generation cap")
    ap.add_argument("--suffix", default="",
                    help="appended to the result run name (e.g. _v3 for grammar ablations)")
    ap.add_argument("--grammar", default=DEFAULT_GRAMMAR, choices=list(GRAMMAR_VERSIONS),
                    help="gloss grammar version for the constraint AND the validity metric")
    args = ap.parse_args()

    eval_set = load_split("eval_subset")
    if args.limit:
        eval_set = eval_set[:args.limit]
    vocab = load_vocab()

    model, tok = load_system(args.system)
    batch_size = args.batch_size or (64 if args.system == "bart" else 8) # if only i ever had the money for more vram

    # BART was trained on raw text -> gloss; the QLoRA model on the zero-shot chat prompt.
    if args.system == "bart":
        inputs = [ex["text"] for ex in eval_set]
    else:
        inputs = [format_prompt(tok, ex["text"], shots=[]) for ex in eval_set]

    if args.copy_propn:
        args.constrained = True

    compiled = tok_info = None
    extras_per_ex = None
    if args.constrained:
        # get model's output size
        vocab_size = model.get_output_embeddings().weight.shape[0]
        # query grammar cache
        cache = GrammarCache(vocab, tok, vocab_size=vocab_size, version=args.grammar)
        tok_info = cache.tokenizer_info
        if args.copy_propn:
            # One grammar per sentence (base vocab + that sentence's copyable words)
            vset = set(vocab)
            extras_per_ex = [copy_candidates(ex["text"], vset) for ex in eval_set]
            compiled = [cache.get(e) for e in extras_per_ex]
            n_with = sum(1 for e in extras_per_ex if e)
            print(f"copy-propn: {n_with}/{len(eval_set)} sentences have copy candidates; "
                  f"{cache.n_compiled} unique grammars compiled")
        else:
            compiled = cache.get(())  # no extras: the plain vocabulary grammar (shared)

    preds, stats = generate_batch(
        model, tok, inputs, constrained=args.constrained, compiled_grammar=compiled,
        tokenizer_info=tok_info, max_new_tokens=args.max_new_tokens, batch_size=batch_size,
    )

    refs = [ex["gloss"] for ex in eval_set]
    metrics = compute_metrics(preds, refs, vocab, extra_valid=extras_per_ex,
                              grammar_version=args.grammar)
    print(json.dumps(metrics, indent=2))
    print(json.dumps(stats, indent=2))

    tag = "copy" if args.copy_propn else ("con" if args.constrained else "unc")
    run_name = f"{args.system}_{tag}{args.suffix}"
    if args.limit:
        run_name += f"_limit{args.limit}"
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / f"{run_name}.json", "w", encoding="utf-8") as f:
        json.dump({
            "run": run_name,
            "grammar_version": args.grammar,
            "config": vars(args),
            "metrics": metrics,
            "speed": stats,
            "examples": [
                {"text": ex["text"], "reference": ex["gloss"], "prediction": p}
                for ex, p in zip(eval_set, preds)
            ],
        }, f, ensure_ascii=False, indent=1)
    print(f"Saved results/{run_name}.json")


if __name__ == "__main__":
    main()
