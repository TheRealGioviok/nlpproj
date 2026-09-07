"""Verify xgrammar constrained generation works on this machine with small random models.
Usage: python scripts/smoke_test_grammar.py
"""

import _bootstrap  # noqa: F401

import torch
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

from glosstrans.generate import generate_batch
from glosstrans.grammar import GRAMMAR_VERSIONS, build_ebnf, compile_grammar, coverage_check
from glosstrans.metrics import token_valid

VOCAB = ["X-I", "LIKE", "DESC-BIG", "DOG", "CAT", "GO", "SCHOOL", "X-YOU", "WANT", "EAT"]
MODELS = [
    ("causal-tiny-llama", "hf-internal-testing/tiny-random-LlamaForCausalLM", AutoModelForCausalLM),
    ("seq2seq-tiny-bart", "sshleifer/bart-tiny-random", AutoModelForSeq2SeqLM),
]


def check(name, model, tokenizer, version):
    tag = f"{name}/{version}"
    vocab_size = model.get_output_embeddings().weight.shape[0]
    compiled, tok_info = compile_grammar(VOCAB, tokenizer, vocab_size=vocab_size, version=version)

    cov = coverage_check(compiled, tokenizer, ["X-I LIKE DOG", "X-YOU WANT EAT CAT"])
    print(f"[{tag}] reference coverage: {cov:.0%}")
    # The compositional rule is the only v2/v3 difference: DESC-DOG is legal in v3 only.
    desc_ok = coverage_check(compiled, tokenizer, ["X-I LIKE DESC-DOG"]) == 1.0
    assert desc_ok == (version == "v3"), f"[{tag}] DESC- composition accepted={desc_ok}, expected {version == 'v3'}"

    outs, _ = generate_batch(
        model, tokenizer, ["Translate to ASL gloss: I like big dogs.", "hello"],
        constrained=True, compiled_grammar=compiled, tokenizer_info=tok_info,
        max_new_tokens=12, batch_size=2,
    )
    print(f"[{tag}] constrained outputs: {outs}")
    legal = set(VOCAB) | ({f"DESC-{w}" for w in VOCAB} if version == "v3" else set())
    for o in outs:
        toks = o.split()
        assert toks, f"[{tag}] empty output"
        bad = [t for t in toks if not token_valid(t, VOCAB, version=version)]
        # Random tiny models don's usually emit EOS, so max_new_tokens can cut the last
        # token mid-word; accept a proper prefix of a legal token.
        if bad == [toks[-1]] and any(w.startswith(toks[-1]) for w in legal):
            bad = []
        assert not bad, f"[{tag}] illegal tokens produced: {bad}"
    print(f"[{tag}] OK - every emitted token is legal under grammar {version}")


def main():
    for v in GRAMMAR_VERSIONS:
        print(f"EBNF preview ({v}):\n{build_ebnf(VOCAB, version=v)}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}, torch={torch.__version__}")

    for name, hf_id, auto_cls in MODELS:
        tok = AutoTokenizer.from_pretrained(hf_id)
        if auto_cls is AutoModelForCausalLM:  # same setup as run_eval.load_system
            tok.pad_token = tok.pad_token or tok.eos_token
            tok.padding_side = "left"
        model = auto_cls.from_pretrained(hf_id).to(device)
        for v in GRAMMAR_VERSIONS:
            check(name, model, tok, v)

    print("\nSMOKE TEST PASSED")


if __name__ == "__main__":
    main()
