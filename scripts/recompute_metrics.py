"""Recompute the metrics block of saved results JSONs from their stored predictions.

Usage: python scripts/recompute_metrics.py results\trainfree_shots0_unc.json [...]
       python scripts/recompute_metrics.py --grammar v2 results\bart_unc.json 
"""

import _bootstrap  # noqa: F401
import argparse
import json

from glosstrans.data import load_vocab
from glosstrans.grammar import GRAMMAR_VERSIONS, copy_candidates
from glosstrans.metrics import compute_metrics


def main():
    ap = argparse.ArgumentParser(
        description="Rewrite the metrics block of saved results JSONs in place.")
    ap.add_argument("results", nargs="+", metavar="RESULTS_JSON",
                    help="results/*.json files to update")
    ap.add_argument("--grammar", default=None, choices=list(GRAMMAR_VERSIONS),
                    help="forced grammar version for the validity metric")
    args = ap.parse_args()

    vocab = load_vocab()
    vset = set(vocab)
    for path in args.results:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        version = args.grammar or d.get("grammar_version")
        if version is None:
            raise SystemExit(f"{path}: no grammar_version in file; pass --grammar v2|v3")
        d["grammar_version"] = d["config"]["grammar"] = version

        preds = [e["prediction"] for e in d["examples"]]
        refs = [e["reference"] for e in d["examples"]]
        extras = None
        if d["config"].get("copy_propn"):
            extras = [copy_candidates(e["text"], vset) for e in d["examples"]]
        old = d["metrics"]
        d["metrics"] = compute_metrics(preds, refs, vocab, extra_valid=extras, grammar_version=version)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(d, f, ensure_ascii=False, indent=1)
        changed = {k: (old.get(k), v) for k, v in d["metrics"].items() if old.get(k) != v}
        print(f"{path} [grammar {version}]: updated {changed if changed else '(no change)'}")


if __name__ == "__main__":
    main()
