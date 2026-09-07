"""Build and compile the gloss-vocabulary EBNF grammar with xgrammar."""

import re
import time

# Numbers pattern
NUMBER_RE = re.compile(r"[0-9]+([.,][0-9]+)*")
NUMBER_EBNF = 'number ::= [0-9]+ (("." | ",") [0-9]+)*'

# Grammar versions (decoding-time only; training never sees the grammar):
#   v2: gloss ::= number | word                 
#   v3: gloss ::= number | word | "DESC-" word 
GRAMMAR_VERSIONS = ("v2", "v3")
DEFAULT_GRAMMAR = "v3"


def check_grammar_version(version: str) -> str:
    if version not in GRAMMAR_VERSIONS:
        raise ValueError(f"unknown grammar version {version!r}; choose from {GRAMMAR_VERSIONS}")
    return version

# List of words which have specific GLOSS tokens or are outright dropped by
# the gloss grammar. Important to not include these back with stuff like the copy rule.
STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "nor", "so", "yet", "if", "then",
    "is", "am", "are", "was", "were", "be", "been", "being", "do", "does",
    "did", "done", "have", "has", "had", "having", "will", "would", "shall",
    "should", "can", "could", "may", "might", "must", "of", "in", "on", "at",
    "to", "for", "from", "by", "with", "without", "about", "against",
    "between", "into", "through", "during", "before", "after", "above",
    "below", "up", "down", "out", "off", "over", "under", "again", "further",
    "once", "here", "there", "when", "where", "why", "how", "all", "any",
    "both", "each", "few", "more", "most", "other", "some", "such", "no",
    "not", "only", "own", "same", "than", "too", "very", "just", "as",
    "it", "its", "this", "that", "these", "those", "i", "me", "my", "we",
    "us", "our", "you", "your", "he", "him", "his", "she", "her", "they",
    "them", "their", "what", "which", "who", "whom", "whose", "also",
}


def is_number_token(tok: str) -> bool:
    return NUMBER_RE.fullmatch(tok) is not None


def _is_inflection_of_known(W: str, vocab) -> bool:
    """True if W looks like an inflected variant of an in-vocab lemma
    (SUFFERING vs SUFFER)"""
    stems = []
    # Plurals
    if W.endswith("IES"):
        stems.append(W[:-3] + "Y")
    if W.endswith("ES"):
        stems.append(W[:-2])
    if W.endswith("S"):
        stems.append(W[:-1])
    # -ing forms
    if W.endswith("ING"):
        stems += [W[:-3], W[:-3] + "E"]
    # participle
    if W.endswith("ED"):
        stems += [W[:-2], W[:-1]]
    return any(s in vocab for s in stems)


def copy_candidates(text: str, vocab) -> tuple:
    """Copy-rule candidates: source words (uppercased) that are
    plausibly open-class passthrough tokens (proper nouns, unseen words).
    skip stopwords, in-vocab words, inflections
    of in-vocab lemmas, words whose DESC-/X- form is in vocab.
    """
    out = []
    for w in text.split():
        W = re.sub(r"[^A-Z0-9'-]", "", w.upper())
        if len(W) < 2 or not W[0].isalpha():
            continue
        if W in vocab or is_number_token(W) or W.lower() in STOPWORDS:
            continue
        if _is_inflection_of_known(W, vocab):
            continue
        if f"DESC-{W}" in vocab or f"X-{W}" in vocab:
            continue
        if W not in out:
            out.append(W)
    return tuple(sorted(out))


def build_ebnf(vocab, extras=(), version=DEFAULT_GRAMMAR):
    """Gloss grammar: gloss tokens separated by single spaces.
    `extras` are per-sentence additional legal tokens (stuff from
    copy rule).

    v2: a token is an enumerated word or a pattern-matched number.
    v3: additionally "DESC-" + any legal word."""
    check_grammar_version(version)

    def esc(tok: str) -> str:
        return tok.replace("\\", "\\\\").replace('"', '\\"')

    words = [t for t in vocab if not is_number_token(t)] + [t for t in extras if t not in vocab]
    alts = " | ".join(f'"{esc(t)}"' for t in words)
    # Optional leading space for SP/BPE tokenizers, which mark word starts with a space
    gloss_rule = 'gloss ::= number | word' + (' | "DESC-" word' if version == "v3" else "")
    return (f'root ::= " "? gloss (" " gloss)*\n'
            f'{gloss_rule}\n'
            f'{NUMBER_EBNF}\n'
            f'word ::= {alts}\n')


class GrammarCache:
    """Compiles the base grammar once and per-sentence extras-variants on
    demand, cached by the extras tuple"""

    def __init__(self, vocab, tokenizer, vocab_size=None, version=DEFAULT_GRAMMAR):
        import xgrammar as xgr

        self.version = check_grammar_version(version)
        kwargs = {"vocab_size": vocab_size} if vocab_size is not None else {}
        self.tokenizer_info = xgr.TokenizerInfo.from_huggingface(tokenizer, **kwargs)
        self.compiler = xgr.GrammarCompiler(self.tokenizer_info)
        self.vocab = list(vocab)
        self._cache = {}

    def get(self, extras=()):
        key = tuple(extras)
        if key not in self._cache:
            t0 = time.perf_counter()
            self._cache[key] = self.compiler.compile_grammar(
                build_ebnf(self.vocab, key, version=self.version))
            if key == ():
                print(f"Base grammar {self.version} compiled in {time.perf_counter() - t0:.1f}s "
                      f"({len(self.vocab)} gloss tokens, "
                      f"tokenizer vocab {self.tokenizer_info.vocab_size})")
        return self._cache[key]

    @property
    def n_compiled(self):
        return len(self._cache)


def compile_grammar(vocab, tokenizer, vocab_size=None, verbose=True, version=DEFAULT_GRAMMAR):
    """Compile the base gloss grammar for a specific HF tokenizer."""
    cache = GrammarCache(vocab, tokenizer, vocab_size=vocab_size, version=version)
    return cache.get(()), cache.tokenizer_info


def matcher_accepts(compiled_grammar, tokenizer, gloss: str) -> bool:
    """Check whether a reference gloss string is fully accepted by the grammar."""
    import xgrammar as xgr

    matcher = xgr.GrammarMatcher(compiled_grammar)
    ids = tokenizer.encode(gloss, add_special_tokens=False)
    for tid in ids:
        if not matcher.accept_token(tid):
            return False
    eos = tokenizer.eos_token_id
    return eos is not None and matcher.accept_token(eos) and matcher.is_terminated()


def coverage_check(compiled_grammar, tokenizer, references, limit=500):
    """Fraction of reference glosses fully accepted token-by-token by the grammar."""
    ok = 0
    n = min(limit, len(references))
    for ref in references[:n]:
        if matcher_accepts(compiled_grammar, tokenizer, ref):
            ok += 1
    return ok / max(n, 1)
