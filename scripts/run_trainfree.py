"""Train-free baseline: zero-shot / few-shot quantized LLM, +/- xgrammar constraint.

Usage:
  python scripts/run_trainfree.py --shots 0 [--constrained] [--limit 50]
  python scripts/run_trainfree.py --shots 5 --constrained
"""

import _bootstrap  # noqa: F401
import argparse
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from glosstrans import RESULTS_DIR
from glosstrans.data import load_split, load_vocab
from glosstrans.generate import generate_batch
from glosstrans.grammar import DEFAULT_GRAMMAR, GRAMMAR_VERSIONS, GrammarCache, copy_candidates
from glosstrans.metrics import compute_metrics
from glosstrans.prompts import format_prompt, pick_fewshot_examples

DEFAULT_MODEL = "unsloth/Llama-3.2-3B-Instruct"

def load_quantized(model_name):
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=bnb, device_map="auto", dtype=torch.bfloat16
    )
    model.eval()
    return model, tok


def main():
    ap = argparse.ArgumentParser(
        description="Run the train-free (prompted, 4-bit LLM) baseline on data/eval_subset.jsonl "
                    "and save metrics + per-example outputs to results/trainfree_shotsK_<unc|con|copy>.json.")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="HF model id to load in 4-bit (default: ungated Llama-3.2-3B-Instruct mirror)")
    ap.add_argument("--shots", type=int, default=5, choices=[0, 5],
                    help="few-shot examples in the prompt: 0 = zero-shot, 5 = fixed seed-42 train examples")
    ap.add_argument("--constrained", action="store_true",
                    help="decode with the xgrammar gloss-vocabulary constraint (default: unconstrained)")
    ap.add_argument("--copy-propn", action="store_true",
                    help="constrained + per-sentence copy rule for proper nouns/unseen words (implies --constrained)")
    ap.add_argument("--limit", type=int, default=None,
                    help="evaluate only the first N examples")
    ap.add_argument("--batch-size", type=int, default=8,
                    help="generation batch size")
    ap.add_argument("--max-new-tokens", type=int, default=128,
                    help="generation cap")
    ap.add_argument("--grammar", default=DEFAULT_GRAMMAR, choices=list(GRAMMAR_VERSIONS),
                    help="gloss grammar version for the constraint AND the validity metric")
    args = ap.parse_args()

    eval_set = load_split("eval_subset")
    if args.limit:
        eval_set = eval_set[:args.limit]
    vocab = load_vocab()
    train = load_split("train")
    shots = pick_fewshot_examples(train, k=args.shots) if args.shots else []

    model, tok = load_quantized(args.model)

    if args.copy_propn:
        args.constrained = True

    compiled = tok_info = None
    extras_per_ex = None
    if args.constrained:
        vocab_size = model.get_output_embeddings().weight.shape[0]
        cache = GrammarCache(vocab, tok, vocab_size=vocab_size, version=args.grammar)
        tok_info = cache.tokenizer_info
        if args.copy_propn:
            vset = set(vocab)
            extras_per_ex = [copy_candidates(ex["text"], vset) for ex in eval_set]
            compiled = [cache.get(e) for e in extras_per_ex]
            n_with = sum(1 for e in extras_per_ex if e)
            print(f"copy-propn: {n_with}/{len(eval_set)} sentences have copy candidates; "
                  f"{cache.n_compiled} unique grammars compiled")
        else:
            compiled = cache.get(())

    prompts = [format_prompt(tok, ex["text"], shots) for ex in eval_set]
    preds, stats = generate_batch(
        model, tok, prompts, constrained=args.constrained, compiled_grammar=compiled,
        tokenizer_info=tok_info, max_new_tokens=args.max_new_tokens, batch_size=args.batch_size,
    )

    refs = [ex["gloss"] for ex in eval_set]
    metrics = compute_metrics(preds, refs, vocab, extra_valid=extras_per_ex,
                              grammar_version=args.grammar)
    print(json.dumps(metrics, indent=2))
    print(json.dumps(stats, indent=2))

    tag = "copy" if args.copy_propn else ("con" if args.constrained else "unc")
    run_name = f"trainfree_shots{args.shots}_{tag}"
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
