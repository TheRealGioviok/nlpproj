"""Unified generation, with or without the xgrammar mask, for both type of models"""

import time

import torch
from transformers import LogitsProcessor


class XGrammarSeq2SeqLogitsProcessor(LogitsProcessor):
    """Grammar mask for encoder-decoder models (BART).

    xgrammar processor assumes a causal LM. BART's decoder requires forced 
    special tokens (</s> <s>) that the grammar would reject, so this
    processor ignores the first `prefix_len` decoder tokens and applies the
    grammar to everything after them.
    """

    def __init__(self, compiled_grammar, tokenizer_info, prefix_len, skip_token_ids=()):
        import xgrammar as xgr

        self._xgr = xgr
        # One grammar for the whole batch, or a list with one per sentence (copy rule).
        self.compiled = compiled_grammar
        self.vocab_size = tokenizer_info.vocab_size
        self.prefix_len = prefix_len
        self.skip = set(t for t in skip_token_ids if t is not None)
        self.matchers = None
        self.bitmask = None

    def _grammar_for(self, i):
        return self.compiled[i] if isinstance(self.compiled, list) else self.compiled

    def __call__(self, input_ids, scores):
        xgr = self._xgr
        bs, cur_len = input_ids.shape
        if cur_len < self.prefix_len:
            return scores  # still emitting forced/special prefix tokens
        if self.matchers is None:
            self.matchers = [xgr.GrammarMatcher(self._grammar_for(i)) for i in range(bs)]
            self.bitmask = xgr.allocate_token_bitmask(bs, self.vocab_size)
        elif cur_len > self.prefix_len:
            for i, m in enumerate(self.matchers):
                tok = input_ids[i, -1].item()
                if tok not in self.skip and not m.is_terminated():
                    m.accept_token(tok)
        for i, m in enumerate(self.matchers):
            if not m.is_terminated():
                m.fill_next_token_bitmask(self.bitmask, i)
        xgr.apply_token_bitmask_inplace(scores, self.bitmask.to(scores.device))
        return scores


def make_grammar_processor(model, tokenizer, compiled_grammar, tokenizer_info):
    """Build a new grammar processor."""
    import xgrammar as xgr

    if model.config.is_encoder_decoder:
        prefix_len = 1 + (1 if getattr(model.config, "forced_bos_token_id", None) is not None else 0)
        skip = {tokenizer.pad_token_id, tokenizer.eos_token_id, tokenizer.bos_token_id}
        return XGrammarSeq2SeqLogitsProcessor(compiled_grammar, tokenizer_info, prefix_len, skip)
    return xgr.contrib.hf.LogitsProcessor(compiled_grammar)


@torch.inference_mode()
def generate_batch(model, tokenizer, inputs, *, constrained=False, compiled_grammar=None,
                   tokenizer_info=None, max_new_tokens=128, batch_size=8):
    """Greedy-decode a list of inputs in batches; returns (texts, speed stats).
    Returns the generated texts and speed statistics.
    """
    is_seq2seq = model.config.is_encoder_decoder
    outputs = []
    total_new_tokens = 0
    t_start = time.perf_counter()

    for i in range(0, len(inputs), batch_size):
        chunk = inputs[i:i + batch_size]
        enc = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True,
                        max_length=1024).to(model.device)
        logits_processor = None
        if constrained:
            cg = compiled_grammar[i:i + batch_size] if isinstance(compiled_grammar, list) else compiled_grammar
            logits_processor = [make_grammar_processor(model, tokenizer, cg, tokenizer_info)]
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            logits_processor=logits_processor,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
        if is_seq2seq:
            new_tokens = out
            texts = tokenizer.batch_decode(out, skip_special_tokens=True)
        else:
            new_tokens = out[:, enc["input_ids"].shape[1]:]
            texts = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        total_new_tokens += int((new_tokens != (tokenizer.pad_token_id or -1)).sum())
        outputs.extend(t.strip() for t in texts)

    elapsed = time.perf_counter() - t_start
    stats = {
        "elapsed_s": elapsed,
        "n_examples": len(inputs),
        "total_new_tokens": total_new_tokens,
        "tokens_per_s": total_new_tokens / elapsed if elapsed > 0 else 0.0,
        "s_per_example": elapsed / len(inputs) if inputs else 0.0,
    }
    return outputs, stats
