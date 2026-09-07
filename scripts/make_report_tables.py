"""LaTeX tables from results/*.json"""

import _bootstrap  # noqa: F401
import argparse
import json
import random
from pathlib import Path

import sacrebleu

import glosstrans
from glosstrans.data import load_vocab
from glosstrans.metrics import norm, token_valid

OUT = Path(__file__).resolve().parent.parent / "report_tables"
OUT.mkdir(exist_ok=True)

SYSTEMS = [  # (label, unc file, con file (v3), copy file (v3))
    ("Zero-shot Llama 3B", "trainfree_shots0_unc", "trainfree_shots0_con", "trainfree_shots0_copy"),
    ("Few-shot Llama 3B ($k{=}5$)", "trainfree_shots5_unc", "trainfree_shots5_con", "trainfree_shots5_copy"),
    ("Fine-tuned BART-base", "bart_unc", "bart_con_v3", "bart_copy_v3"),
    ("QLoRA Llama 3B", "llama-qlora_unc", "llama-qlora_con_v3", "llama-qlora_copy_v3"),
]


RESULTS_DIR = glosstrans.RESULTS_DIR


def load(name):
    with open(RESULTS_DIR / f"{name}.json", encoding="utf-8") as f:
        return json.load(f)


def fmt(x, nd=1):
    return f"{x:.{nd}f}"


def main_grid():
    lines = []
    for label, *files in SYSTEMS:
        runs = [load(f) for f in files]
        cols = ["bleu", "chrf", "rouge_l", "exact_match", "token_validity", "sequence_validity"]
        best = {c: max(r["metrics"][c] for r in runs) for c in cols[:4]}
        for i, (cond, r) in enumerate(zip(["unc", "con", "copy"], runs)):
            m, sp = r["metrics"], r["speed"]
            cells = []
            for c in cols:
                v = m[c]
                s = fmt(v, 2 if c in ("bleu", "chrf", "rouge_l") else 1)
                if c in best and v == best[c]:
                    s = r"\textbf{" + s + "}"
                cells.append(s)
            cells.append(fmt(sp["tokens_per_s"], 0))
            first = (r"\multirow{3}{*}{" + label + "}") if i == 0 else ""
            lines.append(f"{first} & {cond} & " + " & ".join(cells) + r" \\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    body = "\n".join(lines)
    tex = (r"""\begin{tabular}{llrrrrrrr}
\toprule
System & Cond. & BLEU & chrF & ROUGE-L & Exact & Tok.\ val. & Seq.\ val. & tok/s \\
\midrule
""" + body + "\n" + r"\end{tabular}" + "\n")
    (OUT / "main_grid.tex").write_text(tex, encoding="utf-8")


def grammar_ablation():
    """BART / QLoRA under v2 vs v3 (con and copy), plus exact-match split by v2-reachability."""
    vocab = set(load_vocab())
    refs = [e["reference"] for e in load("bart_unc")["examples"]]
    reach = [all(token_valid(t, vocab, version="v2") for t in norm(r).split()) for r in refs]
    n_reach = sum(reach)
    rows = []
    for label, sysname in [("Fine-tuned BART", "bart"), ("QLoRA Llama 3B", "llama-qlora")]:
        variants = [("unc", f"{sysname}_unc"), ("con v2", f"{sysname}_con"), ("con v3", f"{sysname}_con_v3"),
                    ("copy v2", f"{sysname}_copy"), ("copy v3", f"{sysname}_copy_v3")]
        for i, (cond, f) in enumerate(variants):
            d = load(f)
            m = d["metrics"]
            ok = [norm(e["prediction"]) == norm(e["reference"]) for e in d["examples"]]
            er = sum(o for o, r in zip(ok, reach) if r)
            eu = sum(o for o, r in zip(ok, reach) if not r)
            first = (r"\multirow{5}{*}{" + label + "}") if i == 0 else ""
            rows.append(f"{first} & {cond} & {fmt(m['bleu'], 2)} & {fmt(m['chrf'], 2)} & {fmt(m['exact_match'], 1)} & "
                        f"{fmt(m['sequence_validity'], 1)} & {er} & {eu} \\\\")
        rows.append(r"\midrule")
    rows[-1] = r"\bottomrule"
    header = (f"System & Cond. & BLEU & chrF & Exact & Seq.\\ val. & Exact reach.\\ (/{n_reach}) "
              f"& Exact unreach.\\ (/{1000 - n_reach}) \\\\")
    tex = ("\\begin{tabular}{llrrrrrr}\n\\toprule\n" + header + "\n\\midrule\n"
           + "\n".join(rows) + "\n\\end{tabular}\n")
    (OUT / "grammar_ablation.tex").write_text(tex, encoding="utf-8")
    return n_reach


def bootstrap():
    refs = [norm(e["reference"]) for e in load("bart_unc")["examples"]]
    random.seed(0)
    idxs = [[random.randrange(1000) for _ in range(1000)] for _ in range(1000)]

    def preds(f):
        return [norm(e["prediction"]) for e in load(f)["examples"]]

    def boot(A, B):
        db, de = [], []
        for idx in idxs:
            rs = [refs[i] for i in idx]
            db.append(sacrebleu.corpus_bleu([A[i] for i in idx], [rs], force=True).score
                      - sacrebleu.corpus_bleu([B[i] for i in idx], [rs], force=True).score)
            de.append(100 * (sum(A[i] == refs[i] for i in idx) - sum(B[i] == refs[i] for i in idx)) / 1000)
        db.sort()
        de.sort()
        return (db[25], db[975]), (de[25], de[975])

    pairs = [
        ("Few-shot: unc $-$ con", "trainfree_shots5_unc", "trainfree_shots5_con"),
        ("BART: unc $-$ con", "bart_unc", "bart_con_v3"),
        ("BART: con $-$ copy", "bart_con_v3", "bart_copy_v3"),
        ("QLoRA: unc $-$ con", "llama-qlora_unc", "llama-qlora_con_v3"),
        ("QLoRA: unc $-$ copy", "llama-qlora_unc", "llama-qlora_copy_v3"),
        ("QLoRA: copy v3 $-$ copy v2", "llama-qlora_copy_v3", "llama-qlora_copy"),
    ]
    rows = []
    summary = {}
    for name, a, b in pairs:
        (bl, bh), (el, eh) = boot(preds(a), preds(b))
        rows.append(f"{name} & [{bl:+.2f}, {bh:+.2f}] & [{el:+.1f}, {eh:+.1f}] \\\\")
        summary[name] = ((bl, bh), (el, eh))
    header = r"A $-$ B & $\Delta$BLEU CI & $\Delta$Exact CI \\"
    tex = ("\\begin{tabular}{lcc}\n\\toprule\n" + header + "\n\\midrule\n"
           + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    (OUT / "bootstrap.tex").write_text(tex, encoding="utf-8")
    return summary


def hybrid():
    vocab = set(load_vocab())
    refs = [norm(e["reference"]) for e in load("bart_unc")["examples"]]

    def preds(f):
        return [norm(e["prediction"]) for e in load(f)["examples"]]

    def valid(p):
        toks = p.split()
        return bool(toks) and all(token_valid(t, vocab, version="v3") for t in toks)

    rows = []
    for label, u, c in [("BART, con", "bart_unc", "bart_con_v3"),
                        ("BART, copy", "bart_unc", "bart_copy_v3"),
                        ("QLoRA, con", "llama-qlora_unc", "llama-qlora_con_v3"),
                        ("QLoRA, copy", "llama-qlora_unc", "llama-qlora_copy_v3")]:
        U, C = preds(u), preds(c)
        mixed = [a if valid(a) else b for a, b in zip(U, C)]
        hb = sacrebleu.corpus_bleu(mixed, [refs], force=True).score
        he = 100 * sum(p == r for p, r in zip(mixed, refs)) / 1000
        cb = sacrebleu.corpus_bleu(C, [refs], force=True).score
        ce = 100 * sum(p == r for p, r in zip(C, refs)) / 1000
        rows.append(f"{label} & {sum(not valid(a) for a in U)} & {hb:.2f} & {he:.1f} & {cb:.2f} & {ce:.1f} \\\\")
    header = (r"& & \multicolumn{2}{c}{Hybrid} & \multicolumn{2}{c}{Always constrained} \\" + "\n"
              r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}" + "\n"
              r"System, fallback & Fallb. & BLEU & Exact & BLEU & Exact \\")
    tex = ("\\begin{tabular}{lrrrrr}\n\\toprule\n" + header + "\n\\midrule\n"
           + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    (OUT / "hybrid.tex").write_text(tex, encoding="utf-8")


def main():
    global RESULTS_DIR
    ap = argparse.ArgumentParser(description="Generate report_tables/*.tex from results/*.json.")
    ap.add_argument("--results", default=None, help="alternative results directory (default: results/)")
    args = ap.parse_args()
    if args.results:
        RESULTS_DIR = Path(args.results)
    main_grid()
    n_reach = grammar_ablation()
    hybrid()
    summary = bootstrap()
    print("reachable under v2:", n_reach)
    for k, v in summary.items():
        print(k, v)
    for label, *files in SYSTEMS:
        for cond, f in zip(["unc", "con", "copy"], files):
            m = load(f)["metrics"]
            print(f"{label:28s} {cond:4s} BLEU {m['bleu']:6.2f} chrF {m['chrf']:6.2f} R-L {m['rouge_l']:6.2f} "
                  f"EM {m['exact_match']:5.1f} tokv {m['token_validity']:5.1f} seqv {m['sequence_validity']:5.1f} "
                  f"gv={m.get('grammar_version')}")
    print("wrote", sorted(p.name for p in OUT.glob("*.tex")))


if __name__ == "__main__":
    main()
