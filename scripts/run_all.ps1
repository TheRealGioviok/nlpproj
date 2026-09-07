# End-to-end pipeline: setup -> data -> grammar -> baselines -> fine-tunes -> evals -> report.
#
# Usage:  .\scripts\run_all.ps1            full speed (GPU dedicated to this job)
#         .\scripts\run_all.ps1 -Gentle    shared GPU: BelowNormal priority + batch 1 for the
#                                          train-free LLM runs (the setting used for the reported
#                                          prompted-model numbers; fine-tune/eval batches unchanged)

param([switch]$Gentle)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..
if ($Gentle) {
    (Get-Process -Id $PID).PriorityClass = "BelowNormal"
    $TF_BATCH = 1
    Write-Host "Gentle mode: BelowNormal priority, train-free batch size $TF_BATCH" -ForegroundColor Yellow
} else {
    $TF_BATCH = 8
}
$GRAMMAR = "v3"   # main-grid grammar; the v2 ablation steps below pass --grammar v2 explicitly
New-Item -ItemType Directory -Force logs, results, models | Out-Null

# Fix to avoid stderr into stdout or else powershell does windows things
# I am also at my wits end with windows
function Step($name, $log, [string[]]$argv) {
    Write-Host "`n=== $name ===" -ForegroundColor Cyan
    $quoted = $argv | ForEach-Object { if ($_ -match '[\s"]') { '"' + ($_ -replace '"', '\"') + '"' } else { $_ } }
    $cmdline = "`"$PY`" " + ($quoted -join " ") + " 2>&1"
    "$(Get-Date -Format s) START $name" | Add-Content logs\run_all.log
    & cmd /c $cmdline | Tee-Object -FilePath "logs\$log.log"
    $code = $LASTEXITCODE
    "$(Get-Date -Format s) END   $name exit=$code" | Add-Content logs\run_all.log
    if ($code -ne 0) { throw "Step failed: $name (exit $code); see logs\$log.log" }
}

# environ
if (-not (Test-Path .venv)) { py -3.12 -m venv .venv }
$PY = (Resolve-Path .\.venv\Scripts\python.exe).Path
Step "pip: torch cu128"    "setup_torch"   @("-m", "pip", "install", "torch", "--index-url", "https://download.pytorch.org/whl/cu128")
Step "pip: requirements"   "setup_reqs"    @("-m", "pip", "install", "-r", "requirements.txt")
Step "pip: triton-windows (xgrammar CUDA bitmask kernel; plain triton has no Windows wheel)" "setup_triton" @("-m", "pip", "install", "triton-windows")
Step "register notebook kernel nlpproj-venv" "setup_kernel" @("-m", "ipykernel", "install", "--user", "--name", "nlpproj-venv")
Step "smoke test xgrammar (causal + BART seq2seq processor)" "smoke_test" @("scripts\smoke_test_grammar.py")

# data prep
# Kaggle CSV downloads anonymously via kagglehub. If that fails, download manually and run
# scripts\prepare_data.py --csv path\to\file.csv by hand, then relaunch.
Step "prepare data (dedupe -> 80/10/10 seed 42 -> eval_subset + vocab.txt)" "prepare_data" @("scripts\prepare_data.py")

# grammar coverage numbers
Step "grammar coverage (BART tokenizer)"  "coverage_bart"  @("scripts\check_grammar_coverage.py", "--model", "facebook/bart-base", "--limit", "1000")
Step "grammar coverage (Llama tokenizer)" "coverage_llama" @("scripts\check_grammar_coverage.py", "--model", "unsloth/Llama-3.2-3B-Instruct", "--limit", "1000")

# train-free baselines (batch size from -Gentle; see header)
foreach ($shots in 0, 5) {
    $tf = @("scripts\run_trainfree.py", "--shots", "$shots", "--batch-size", "$TF_BATCH")
    Step "trainfree shots$shots unc"  "trainfree_shots${shots}_unc"  $tf
    Step "trainfree shots$shots con"  "trainfree_shots${shots}_con"  ($tf + @("--grammar", $GRAMMAR, "--constrained"))
    Step "trainfree shots$shots copy" "trainfree_shots${shots}_copy" ($tf + @("--grammar", $GRAMMAR, "--constrained", "--copy-propn"))
}

# BART fine-tune + evals (main grid under v3 with --suffix _v3; v2 ablation without suffix)
Step "train BART (3 ep, eff. batch 32)" "train_bart" @("scripts\train_bart.py", "--epochs", "3", "--batch-size", "16", "--grad-accum", "2")
$ev = @("scripts\run_eval.py", "--system", "bart", "--batch-size", "16")
Step "eval bart unc"                "eval_bart_unc"     $ev
Step "eval bart con  (main, v3)"    "eval_bart_con_v3"  ($ev + @("--grammar", $GRAMMAR, "--suffix", "_v3", "--constrained"))
Step "eval bart copy (main, v3)"    "eval_bart_copy_v3" ($ev + @("--grammar", $GRAMMAR, "--suffix", "_v3", "--constrained", "--copy-propn"))
Step "eval bart con  (v2 ablation)" "eval_bart_con"     ($ev + @("--grammar", "v2", "--constrained"))
Step "eval bart copy (v2 ablation)" "eval_bart_copy"    ($ev + @("--grammar", "v2", "--constrained", "--copy-propn"))

# QLoRA fine-tune + evals
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
Step "train QLoRA (1 ep, 20k pairs, eff. batch 16)" "train_qlora" @("scripts\train_llama_qlora.py", "--epochs", "1", "--batch-size", "2", "--grad-accum", "8", "--max-samples", "20000")
$ev = @("scripts\run_eval.py", "--system", "llama-qlora", "--batch-size", "8")
Step "eval qlora unc"                "eval_qlora_unc"     $ev
Step "eval qlora con  (main, v3)"    "eval_qlora_con_v3"  ($ev + @("--grammar", $GRAMMAR, "--suffix", "_v3", "--constrained"))
Step "eval qlora copy (main, v3)"    "eval_qlora_copy_v3" ($ev + @("--grammar", $GRAMMAR, "--suffix", "_v3", "--constrained", "--copy-propn"))
Step "eval qlora con  (v2 ablation)" "eval_qlora_con"     ($ev + @("--grammar", "v2", "--constrained"))
Step "eval qlora copy (v2 ablation)" "eval_qlora_copy"    ($ev + @("--grammar", "v2", "--constrained", "--copy-propn"))

# Metrics, notebook, tables
$files = Get-ChildItem results\*.json | Where-Object { $_.Name -notmatch "_limit" } | ForEach-Object FullName
Step "recompute metrics on every result (same metric defs everywhere)" "recompute_metrics" (@("scripts\recompute_metrics.py") + $files)
Step "execute report.ipynb" "nbconvert" @("-m", "jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace", "--ExecutePreprocessor.timeout=-1", "--ExecutePreprocessor.kernel_name=nlpproj-venv", "report.ipynb")
Step "generate report tables (report_tables\*.tex)" "make_report_tables" @("scripts\make_report_tables.py")

Write-Host "`nALL DONE. Results in results\*.json, tables in report_tables\ (latexmk -pdf report.tex), charts in report.ipynb." -ForegroundColor Green
