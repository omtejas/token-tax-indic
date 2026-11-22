$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$launcher = Get-Command py -ErrorAction SilentlyContinue
if ($launcher) { py -3.14 -m venv .venv } else { python -m venv .venv }
if ($LASTEXITCODE -ne 0) { throw "Could not create the Python environment. Python 3.14 is recommended." }
$python = ".\.venv\Scripts\python.exe"
& $python -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed." }
& $python -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw "Project dependency installation failed." }
Write-Host "Installing the CUDA 13.0 PyTorch wheel for your NVIDIA driver..."
& $python -m pip install torch --index-url https://download.pytorch.org/whl/cu130
if ($LASTEXITCODE -ne 0) { throw "CUDA PyTorch installation failed. Check the official PyTorch selector." }
& $python -m pip install -e .
if ($LASTEXITCODE -ne 0) { throw "Editable project installation failed." }
& $python -m pytest -q
if ($LASTEXITCODE -ne 0) { throw "Unit tests failed." }
& $python -c "import torch, sys; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No CUDA GPU'); sys.exit(0 if torch.cuda.is_available() else 1)"
if ($LASTEXITCODE -ne 0) { throw "PyTorch cannot access the NVIDIA GPU. Review the CUDA wheel and driver." }
Write-Host "Setup complete. Run .\run_experiments.ps1 to generate research measurements."
