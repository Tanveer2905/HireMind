"""
download_models.py — Model Download Script
Downloads and saves all required models to the local /models directory.
Run this once during setup; after that, the system works fully offline.
"""

import os
import sys
import io
import shutil
from pathlib import Path

# Fix Windows console encoding for Unicode characters
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"
BGE_SMALL_PATH = MODELS_DIR / "bge-small-en"
BGE_BASE_PATH = MODELS_DIR / "bge-base-en-v1.5"
SPACY_MODEL_PATH = MODELS_DIR / "en_core_web_sm"
LLAMA3_MODEL_PATH = MODELS_DIR / "llama3-8b-instruct-q4_0.gguf"


def download_bge_base_model():
    """Download BAAI/bge-base-en-v1.5 (upgraded model) and save locally."""
    print()
    print("=" * 60)
    print("  Downloading BAAI/bge-base-en-v1.5 embedding model...")
    print("=" * 60)
    print()

    if BGE_BASE_PATH.exists() and any(BGE_BASE_PATH.iterdir()):
        print(f"  ✅ Model already exists at {BGE_BASE_PATH}")
        print("     (Delete the folder to re-download)")
        return

    try:
        from sentence_transformers import SentenceTransformer

        print(f"  Downloading to: {BGE_BASE_PATH}")
        print("  This may take a few minutes (~440MB)...")
        print()

        # Download model from HuggingFace
        model = SentenceTransformer("BAAI/bge-base-en-v1.5")

        # Save to local directory
        BGE_BASE_PATH.mkdir(parents=True, exist_ok=True)
        model.save(str(BGE_BASE_PATH))

        print(f"  ✅ Model saved to {BGE_BASE_PATH}")

        # Verify we can load it back
        test_model = SentenceTransformer(str(BGE_BASE_PATH))
        test_embedding = test_model.encode(["test"], normalize_embeddings=True)
        print(f"  ✅ Model verified (dim={test_embedding.shape[1]})")

    except Exception as e:
        print(f"  ⚠️  Failed to download bge-base-en-v1.5: {e}")
        print("     Falling back to bge-small-en...")
        download_bge_small_model()


def download_bge_small_model():
    """Download BAAI/bge-small-en (fallback model) and save locally."""
    print()
    print("=" * 60)
    print("  Downloading BAAI/bge-small-en embedding model (fallback)...")
    print("=" * 60)
    print()

    if BGE_SMALL_PATH.exists() and any(BGE_SMALL_PATH.iterdir()):
        print(f"  ✅ Model already exists at {BGE_SMALL_PATH}")
        return

    try:
        from sentence_transformers import SentenceTransformer

        print(f"  Downloading to: {BGE_SMALL_PATH}")
        print("  This may take a few minutes (~130MB)...")
        print()

        model = SentenceTransformer("BAAI/bge-small-en")
        BGE_SMALL_PATH.mkdir(parents=True, exist_ok=True)
        model.save(str(BGE_SMALL_PATH))

        print(f"  ✅ Model saved to {BGE_SMALL_PATH}")

        test_model = SentenceTransformer(str(BGE_SMALL_PATH))
        test_embedding = test_model.encode(["test"], normalize_embeddings=True)
        print(f"  ✅ Model verified (dim={test_embedding.shape[1]})")

    except Exception as e:
        print(f"  ❌ Failed to download BGE model: {e}")
        sys.exit(1)


def download_spacy_model():
    """Download spaCy en_core_web_sm and save to local models directory."""
    print()
    print("=" * 60)
    print("  Downloading spaCy en_core_web_sm model...")
    print("=" * 60)
    print()

    if SPACY_MODEL_PATH.exists() and any(SPACY_MODEL_PATH.iterdir()):
        print(f"  ✅ Model already exists at {SPACY_MODEL_PATH}")
        print("     (Delete the folder to re-download)")
        return

    try:
        import spacy
        from spacy.cli import download

        # Download the model via spaCy CLI
        print("  Downloading en_core_web_sm (~12MB)...")
        download("en_core_web_sm")

        # Find where spaCy installed it
        nlp = spacy.load("en_core_web_sm")
        source_path = Path(nlp.path)

        # Copy to our local models directory
        SPACY_MODEL_PATH.mkdir(parents=True, exist_ok=True)

        if source_path.exists():
            # Copy the model directory contents
            for item in source_path.iterdir():
                dest = SPACY_MODEL_PATH / item.name
                if item.is_dir():
                    if dest.exists():
                        shutil.rmtree(dest)
                    shutil.copytree(item, dest)
                else:
                    shutil.copy2(item, dest)

            print(f"  ✅ Model saved to {SPACY_MODEL_PATH}")
        else:
            print(f"  ⚠️  Could not find model source at {source_path}")
            print("     The model is installed in the venv and will work from there.")

        # Verify
        test_nlp = spacy.load(str(SPACY_MODEL_PATH))
        doc = test_nlp("This is a test sentence.")
        print(f"  ✅ Model verified ({len(doc)} tokens)")

    except Exception as e:
        print(f"  ❌ Failed to download spaCy model: {e}")
        print("     Trying fallback method...")
        try:
            import subprocess
            subprocess.check_call([
                sys.executable, "-m", "spacy", "download", "en_core_web_sm"
            ])
            print("  ✅ spaCy model installed in venv (fallback)")
        except Exception as e2:
            print(f"  ❌ Fallback also failed: {e2}")
            sys.exit(1)


def download_gguf_model():
    """Download LLaMA 3 8B Instruct GGUF model."""
    print()
    print("=" * 60)
    print("  Downloading LLaMA 3 8B Instruct (GGUF)...")
    print("=" * 60)
    print()

    if LLAMA3_MODEL_PATH.exists():
        print(f"  ✅ Model already exists at {LLAMA3_MODEL_PATH}")
        return

    try:
        from huggingface_hub import hf_hub_download
        print(f"  Downloading to: {LLAMA3_MODEL_PATH}")
        print("  This may take a while (~4.7GB)...")
        print()

        hf_hub_download(
            repo_id="QuantFactory/Meta-Llama-3-8B-Instruct-GGUF",
            filename="Meta-Llama-3-8B-Instruct.Q4_0.gguf",
            local_dir=str(MODELS_DIR),
            local_dir_use_symlinks=False
        )

        # Rename to expected filename
        downloaded_file = MODELS_DIR / "Meta-Llama-3-8B-Instruct.Q4_0.gguf"
        if downloaded_file.exists():
            downloaded_file.rename(LLAMA3_MODEL_PATH)

        print(f"  ✅ Model saved to {LLAMA3_MODEL_PATH}")

    except Exception as e:
        print(f"  ❌ Failed to download LLaMA 3 model: {e}")
        print("     To manually download:")
        print("     1. Go to https://huggingface.co/QuantFactory/Meta-Llama-3-8B-Instruct-GGUF/tree/main")
        print("     2. Download Meta-Llama-3-8B-Instruct.Q4_0.gguf")
        print("     3. Place it in the models/ directory and rename to llama3-8b-instruct-q4_0.gguf")


def verify_setup():
    """Verify all models are ready."""
    print()
    print("=" * 60)
    print("  Verifying setup...")
    print("=" * 60)
    print()

    all_ok = True

    # Check BGE model (base preferred, small as fallback)
    if BGE_BASE_PATH.exists() and any(BGE_BASE_PATH.iterdir()):
        print(f"  ✅ BGE model (base):  {BGE_BASE_PATH}")
    elif BGE_SMALL_PATH.exists() and any(BGE_SMALL_PATH.iterdir()):
        print(f"  ✅ BGE model (small): {BGE_SMALL_PATH}")
    else:
        print(f"  ❌ BGE model: NOT FOUND")
        all_ok = False

    # Check spaCy model
    if SPACY_MODEL_PATH.exists() and any(SPACY_MODEL_PATH.iterdir()):
        print(f"  ✅ spaCy model: {SPACY_MODEL_PATH}")
    else:
        # Check if available in venv
        try:
            import spacy
            spacy.load("en_core_web_sm")
            print(f"  ✅ spaCy model: available in venv")
        except Exception:
            print(f"  ❌ spaCy model: NOT FOUND")
            all_ok = False

    # Check FAISS
    try:
        import faiss
        print(f"  ✅ FAISS:       v{faiss.__version__ if hasattr(faiss, '__version__') else 'installed'}")
    except ImportError:
        print(f"  ❌ FAISS:       NOT INSTALLED")
        all_ok = False

    # Check pdfplumber
    try:
        import pdfplumber
        print(f"  ✅ pdfplumber:  v{pdfplumber.__version__}")
    except ImportError:
        print(f"  ❌ pdfplumber:  NOT INSTALLED")
        all_ok = False

    # Check new dependencies
    try:
        import huggingface_hub
        print(f"  ✅ huggingface-hub: v{huggingface_hub.__version__}")
    except ImportError:
        print(f"  ⚠️  huggingface-hub: NOT INSTALLED")

    try:
        import lightgbm
        print(f"  ✅ lightgbm:    v{lightgbm.__version__}")
    except ImportError:
        print(f"  ⚠️  lightgbm:    NOT INSTALLED (needed for feedback learning)")

    # Check llama-cpp-python
    try:
        import llama_cpp
        print(f"  ✅ llama-cpp:   installed")
    except ImportError:
        print(f"  ⚠️  llama-cpp:   NOT INSTALLED")

    # Check GGUF model
    if LLAMA3_MODEL_PATH.exists():
        print(f"  ✅ LLaMA 3 GGUF: {LLAMA3_MODEL_PATH}")
    else:
        print(f"  ⚠️  LLaMA 3 GGUF: NOT FOUND (LLM features will be disabled)")

    print()
    if all_ok:
        print("  ✅ All core checks passed! System is ready.")
    else:
        print("  ❌ Some checks failed. Please review errors above.")

    print()
    return all_ok


if __name__ == "__main__":
    print()
    print("╔" + "═" * 58 + "╗")
    print("║" + "  🤖  AI HIRING COPILOT — Model Setup  ".center(58) + "║")
    print("╚" + "═" * 58 + "╝")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    download_bge_base_model()
    download_spacy_model()
    download_gguf_model()
    success = verify_setup()

    sys.exit(0 if success else 1)
