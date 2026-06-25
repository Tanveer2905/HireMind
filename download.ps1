$ErrorActionPreference = "Stop"
.\venv\Scripts\Activate.ps1
$env:HF_HOME = "$PWD\data\.cache\huggingface"
python download_models.py
