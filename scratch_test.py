import logging
import sys
logging.basicConfig(level=logging.INFO)

try:
    from llama_cpp import Llama
    print("Import successful")
    
    # Try loading with use_mmap=False
    print("Testing load with use_mmap=False...")
    l = Llama(
        model_path='d:/Antigravity/ai_recruiter/models/llama3-8b-instruct-q4_K_M.gguf',
        n_ctx=512,
        use_mmap=False,
        use_mlock=False,
        verbose=True
    )
    print("SUCCESSFUL LOAD!")
except Exception as e:
    print(f"FAILED: {e}")
    sys.exit(1)
