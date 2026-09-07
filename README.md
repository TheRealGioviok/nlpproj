# Constrained vs. Unconstrained Decoding for English → ASL Gloss

Systems (all evaluated on the same 1,000-example test subset, under three decoding
conditions — unconstrained / constrained / constrained + per-sentence copy rule):

1. Zero-shot Llama-3.2-3B-Instruct (4-bit)
2. Few-shot (k=5) Llama-3.2-3B-Instruct (4-bit)
3. Fine-tuned facebook/bart-base
4. QLoRA fine-tuned Llama-3.2-3B-Instruct

The written report is `report.tex` (its tables are generated into `report_tables/` by
`scripts/make_report_tables.py` from `results/*.json`; executing `report.ipynb` writes the
report's charts to `figures/*.pdf` and holds additional diagnostics not in the report; the
three diagrams are TikZ inside `report.tex`).

## Setup (Windows, RTX 3080 10GB, Python 3.12)

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m pip install triton-windows                          # xgrammar's CUDA bitmask kernel (plain triton has no Windows wheel)
.venv\Scripts\python -m ipykernel install --user --name nlpproj-venv       # kernel used to execute report.ipynb
.venv\Scripts\python scripts\smoke_test_grammar.py                          # verify xgrammar works (causal + BART seq2seq processor)
```

A HuggingFace token is not required (the Llama weights come from the ungated mirror
`unsloth/Llama-3.2-3B-Instruct`) but is recommended for rate limits:
`.venv\Scripts\python -c "from huggingface_hub import login; login()"`.

## Reproducing everything with one command

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_all.ps1
```

chains setup → data → coverage checks → the 6 train-free runs → both fine-tunes with
their 6 main-grid evals → the 4 v2-ablation evals → metric recomputation → notebook
execution → report tables. `-Gentle` runs at BelowNormal priority with batch
size 1 for the train-free runs. The step-by-step equivalent follows; activate the venv
first (`.venv\Scripts\Activate.ps1`) or prefix commands with `.venv\Scripts\python`.

## Data

```powershell
python scripts\prepare_data.py
```

downloads the [Kaggle ASLG-PC12 CSV](https://www.kaggle.com/datasets/thedevastator/unlock-the-power-of-english-asl-with-aslg-pc12-c) anonymously via `kagglehub` (no credentials needed). If the download fails, fetch the CSV manually and run `python scripts\prepare_data.py --csv path\to\file.csv`.

Outputs: `data/{train,val,test,eval_subset}.jsonl` (exact-dupe removal **before** the
80/10/10 split, seed 42; `eval_subset` = first 1,000 test pairs) and `data/vocab.txt`
(gloss vocabulary from the train split only).

## Grammar coverage check

```powershell
python scripts\check_grammar_coverage.py --model facebook/bart-base --limit 1000
python scripts\check_grammar_coverage.py --model unsloth/Llama-3.2-3B-Instruct --limit 1000
```

verifies that the compiled grammar accepts validation references exactly when all their
tokens are legal.

## Main grid: 4 systems × {unc, con, copy}

```powershell
# Train-free baselines (6 runs)
foreach ($shots in 0, 5) {
    python scripts\run_trainfree.py --shots $shots --batch-size 1
    python scripts\run_trainfree.py --shots $shots --batch-size 1 --constrained
    python scripts\run_trainfree.py --shots $shots --batch-size 1 --constrained --copy-propn
}

# Fine-tunes (hyperparameters as reported; checkpoints every 500 steps, auto-resume on relaunch)
python scripts\train_bart.py --epochs 3 --batch-size 16 --grad-accum 2
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
python scripts\train_llama_qlora.py --epochs 1 --batch-size 2 --grad-accum 8 --max-samples 20000

# Fine-tuned evals (6 runs).
foreach ($sys in @{n='bart'; b=16}, @{n='llama-qlora'; b=8}) {
    python scripts\run_eval.py --system $sys.n --batch-size $sys.b
    python scripts\run_eval.py --system $sys.n --batch-size $sys.b --grammar v3 --suffix _v3 --constrained
    python scripts\run_eval.py --system $sys.n --batch-size $sys.b --grammar v3 --suffix _v3 --constrained --copy-propn
}
```

Every run script accepts `--limit 50` for a quick smoke run. Results land in
`results/<run>.json` (config + metrics + all 1,000 source/reference/prediction triples).

## Grammar versions (v2 vs. v3)

Every eval script takes `--grammar v2|v3` (default v3); the flag selects both the
decoding grammar and the validity metric, and is recorded in the result JSON
(`grammar_version`).

## Metrics and report

```powershell
# Recompute metrics for all full runs with one consistent code path
python scripts\recompute_metrics.py (Get-ChildItem results\*.json | Where-Object { $_.Name -notmatch "_limit" } | ForEach-Object FullName)

# Execute the companion notebook (writes figures/*.pdf; per-system deltas, coverage ceilings, garden-path evidence)
python -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.kernel_name=nlpproj-venv report.ipynb

# Generate the LaTeX tables of the report from results/*.json (report_tables/*.tex), then build it
python scripts\make_report_tables.py
latexmk -pdf report.tex
```

Metrics: sacreBLEU, chrF, ROUGE-L, exact match, token/sequence gloss validity,
reference OOV rate, tokens/s and s/example. Because every configuration ran once,
differences are assessed with a paired bootstrap (1,000 resamples, seed 0), computed
in `scripts/make_report_tables.py`.

## Pretrained models

Both fine-tunes are published on the Hugging Face Hub, so nothing has to be retrained to try the
demo or re-run the evaluation:

| System | Hub repo | Size | Contents |
|---|---|---|---|
| Fine-tuned BART-base | [425GMM/bart-base-aslg-gloss](https://huggingface.co/425GMM/bart-base-aslg-gloss) | 269 MB | full weights (fp16), tokenizer |
| QLoRA Llama-3.2-3B | [425GMM/llama-3.2-3b-qlora-aslg-gloss](https://huggingface.co/425GMM/llama-3.2-3b-qlora-aslg-gloss) | 109 MB | LoRA adapter only; the base `unsloth/Llama-3.2-3B-Instruct` is fetched automatically |

`scripts/run_eval.py` and `scripts/translate_repl.py` load from `models/` when the local
training output exists and fall back to the Hub repos otherwise, so on a fresh clone the
first run downloads them into the Hugging Face cache. To fetch them explicitly, or to put
them where the training scripts would have written them:

```powershell
.venv\Scripts\hf download 425GMM/bart-base-aslg-gloss --local-dir models\bart-base-aslg
.venv\Scripts\hf download 425GMM/llama-3.2-3b-qlora-aslg-gloss --local-dir models\llama3b-qlora-aslg
```

## Trying the models interactively

```powershell
python scripts\translate_repl.py --models both
```

Needs `data/vocab.txt` (run `scripts\prepare_data.py` once) and either the local models or
network access for the first download.
