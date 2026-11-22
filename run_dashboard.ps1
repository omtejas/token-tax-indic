$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
if (-not (Test-Path .\.venv\Scripts\python.exe)) { throw "Run setup_windows.ps1 first." }
& .\.venv\Scripts\python.exe -m streamlit run app.py
