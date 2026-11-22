$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
if (-not (Get-Command py -ErrorAction SilentlyContinue)) { throw "Python launcher (py) was not found. Install Python 3.14 and enable the launcher." }
py -3.14 -m venv .venv
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
& .\.venv\Scripts\python.exe -m pip install -e .
& .\.venv\Scripts\python.exe -m pytest -q
& .\.venv\Scripts\python.exe -m token_tax.cli gpu-info
Write-Host "Setup complete. Run .\run_dashboard.ps1 to open the dashboard."
