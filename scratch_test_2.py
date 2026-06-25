import logging
import sys
import os

os.environ["GGML_NO_AVX2"] = "1"
os.environ["GGML_NO_FMA"] = "1"
os.environ["GGML_NO_F16C"] = "1"

logging.basicConfig(level=logging.INFO)

try:
    from llama_cpp import Llama
    print("Import successful")
    
    print("Testing load with GGML flags disabled...")
    l = Llama(
        model_path='d:/Antigravity/ai_recruiter/models/llama3-8b-instruct-q4_0.gguf',
        n_ctx=512,
        verbose=True
    )
    print("SUCCESSFUL LOAD!")
except Exception as e:
    print(f"FAILED: {e}")
    sys.exit(1)
