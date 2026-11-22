param(
  [switch]$SkipTokenizer,
  [switch]$SkipGpu,
  [switch]$SkipFailureAnalysis
)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$Python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) { throw "Virtual environment not found. Run .\setup_windows.ps1 first." }
New-Item -ItemType Directory -Force -Path (Join-Path $PSScriptRoot "results") | Out-Null

if (-not $SkipTokenizer) {
  Write-Host "[1/3] Running aligned five-language tokenizer benchmark..."
  & $Python -m token_tax.cli tokenize --config configs\experiment.yaml
  if ($LASTEXITCODE -ne 0) { throw "Tokenizer benchmark failed (exit $LASTEXITCODE)." }
}
if (-not $SkipGpu) {
  Write-Host "[2/3] Running repeated native-model GPU benchmark (3 repetitions)..."
  & $Python -m token_tax.cli gpu-benchmark --config configs\experiment.yaml --model-config configs\model.yaml
  if ($LASTEXITCODE -ne 0) { throw "GPU benchmark failed (exit $LASTEXITCODE). Check VRAM and CUDA." }
}
if (-not $SkipFailureAnalysis) {
  $latestTokenizer = Get-ChildItem (Join-Path $PSScriptRoot "results\tokenizer_metrics_*.csv") -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime -Descending | Select-Object -First 1
  if ($latestTokenizer) {
    Write-Host "[3/3] Generating high-fertility failure examples..."
    & $Python -m token_tax.cli failure-analysis --input $latestTokenizer.FullName --output results\failure_examples.csv --top-n 20
    if ($LASTEXITCODE -ne 0) { throw "Failure analysis failed (exit $LASTEXITCODE)." }
  } else {
    Write-Warning "No tokenizer CSV found; skipping failure analysis."
  }
}
Write-Host "Study stages completed. Open results with .\run_dashboard.ps1"
