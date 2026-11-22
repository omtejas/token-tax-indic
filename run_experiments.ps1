$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
if (-not (Test-Path .\.venv\Scripts\python.exe)) { throw "Run setup_windows.ps1 first." }
& .\.venv\Scripts\python.exe -m token_tax.cli tokenize --config configs/experiment.yaml
if ($LASTEXITCODE -ne 0) { throw "Tokenizer benchmark failed." }
& .\.venv\Scripts\python.exe -m token_tax.cli gpu-benchmark --config configs/experiment.yaml --model-config configs/model.yaml
if ($LASTEXITCODE -ne 0) { throw "GPU benchmark failed. Review the error and available VRAM." }
Write-Host "Both experiment stages completed. Results are in .\results."
