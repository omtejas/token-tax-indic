$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$launcher = Get-Command py -ErrorAction SilentlyContinue
if ($launcher) { py -3.14 -m venv .venv } else { python -m venv .venv }
if ($LASTEXITCODE -ne 0) { throw "Could not create the Python environment. Confirm Python 3.14 is installed." }
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
& .\.venv\Scripts\python.exe -m pip install -e .
& .\.venv\Scripts\python.exe -m pytest -q
& .\.venv\Scripts\python.exe -m token_tax.cli gpu-info
if ($LASTEXITCODE -ne 0) { throw "Environment setup or GPU check failed." }
Write-Host "Setup complete. Run .\run_experiments.ps1 to generate research measurements."
