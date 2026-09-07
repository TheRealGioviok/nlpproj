"""Evaluation metrics to measure translation quality, gloss validity and speed."""

import re

from .grammar import DEFAULT_GRAMMAR, check_grammar_version, is_number_token

_WS = re.compile(r"\s+")


def norm(s: str) -> str:
    return _WS.sub(" ", s.strip()).upper()


def token_valid(t, vocab, extras=frozenset(), version=DEFAULT_GRAMMAR):
    """Python implementation of the same validity definition as the grammar used for decoding."""
    if t in vocab or t in extras or is_number_token(t):
        return True
    if version == "v3":
        return t.startswith("DESC-") and (t[5:] in vocab or t[5:] in extras)
    return False


def compute_metrics(predictions, references, vocab, extra_valid=None,
                    grammar_version=DEFAULT_GRAMMAR):
    """Full evaluation metrics for a set of predictions and references: BLEU, CHRF, ROUGE-L, exact match, token-level and sequence-level gloss validity, and reference OOV rate."""
    check_grammar_version(grammar_version)
    import sacrebleu
    from rouge_score import rouge_scorer

    preds = [norm(p) for p in predictions]
    refs = [norm(r) for r in references]
    vocab = set(vocab)

    bleu = sacrebleu.corpus_bleu(preds, [refs])
    chrf = sacrebleu.corpus_chrf(preds, [refs])

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    rouge_l = sum(scorer.score(r, p)["rougeL"].fmeasure for p, r in zip(preds, refs)) / max(len(preds), 1)

    exact = sum(p == r for p, r in zip(preds, refs)) / max(len(preds), 1)

    # Gloss validity: token-level and sequence-level
    total_tokens = valid_tokens = 0
    valid_seqs = 0
    for idx, p in enumerate(preds):
        extras = set(extra_valid[idx]) if extra_valid else set()
        toks = p.split()
        total_tokens += len(toks)
        in_vocab = sum(token_valid(t, vocab, extras, version=grammar_version) for t in toks)
        valid_tokens += in_vocab
        if toks and in_vocab == len(toks):
            valid_seqs += 1

    # Reference OOV rate.
    ref_total = ref_oov = 0
    for r in refs:
        toks = r.split()
        ref_total += len(toks)
        ref_oov += sum(not token_valid(t, vocab, version=grammar_version) for t in toks)

    return {
        "bleu": round(bleu.score, 2),
        "chrf": round(chrf.score, 2),
        "rouge_l": round(100 * rouge_l, 2),
        "exact_match": round(100 * exact, 2),
        "token_validity": round(100 * valid_tokens / max(total_tokens, 1), 2),
        "sequence_validity": round(100 * valid_seqs / max(len(preds), 1), 2),
        "ref_oov_rate": round(100 * ref_oov / max(ref_total, 1), 3),
        "grammar_version": grammar_version,
        "n": len(preds),
    }
