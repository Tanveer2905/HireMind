$ErrorActionPreference = "Stop"

Write-Host "Creating Virtual Environment..."
python -m venv venv

Write-Host "Activating Virtual Environment..."
.\venv\Scripts\Activate.ps1

Write-Host "Setting Huggingface cache dir..."
$env:HF_HOME = "$PWD\data\.cache\huggingface"

Write-Host "Upgrading pip..."
python -m pip install --upgrade pip

Write-Host "Installing requirements..."
pip install -r requirements.txt

Write-Host "Installing llama-cpp-python (with CUDA support if available)..."
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu118

Write-Host "Downloading models (This will take a while for the 5GB Llama model)..."
python download_models.py

Write-Host "Starting the backend server..."
python -m backend.main
