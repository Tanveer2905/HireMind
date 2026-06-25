"""
llm_client.py — Embedded LLM Client
Wraps llama-cpp-python to provide local, in-process LLM generation.
Loads the model lazily to conserve RAM until requested.
"""

import json
import logging
import re
import os
import threading

os.environ["GGML_NO_AVX2"] = "1"
os.environ["GGML_NO_FMA"] = "1"
os.environ["GGML_NO_F16C"] = "1"

from utils import LLM_CACHE_PATH, LLAMA3_MODEL_PATH

logger = logging.getLogger(__name__)

# Try to load cache from disk
if LLM_CACHE_PATH.exists():
    try:
        with open(LLM_CACHE_PATH, "r", encoding="utf-8") as f:
            _LLM_CACHE = json.load(f)
    except Exception as e:
        logger.warning(f"Could not load LLM cache: {e}")
        _LLM_CACHE = {}
else:
    _LLM_CACHE = {}

# Global Llama instance (lazy loaded)
_llama_instance = None
_llama_lock = threading.Lock()


class LlamaClient:
    """
    Embedded LLM client using llama-cpp-python.
    Loads the GGUF model directly into the Python process.
    """

    def __init__(self, model_path=str(LLAMA3_MODEL_PATH)):
        self.model_path = model_path
        
    def _get_instance(self):
        """Lazy loads the model."""
        global _llama_instance
        with _llama_lock:
            if _llama_instance is None:
                if not os.path.exists(self.model_path):
                    raise FileNotFoundError(
                        f"LLM Model not found at {self.model_path}. "
                        "Run setup or download_models.py to download it."
                    )
                
                logger.info(f"Loading local LLM into memory: {self.model_path}")
                try:
                    from llama_cpp import Llama
                    # Load with n_ctx=4096 (standard for our prompts) and offload to GPU if available
                    _llama_instance = Llama(
                        model_path=self.model_path,
                        n_ctx=4096,
                        n_gpu_layers=20,  # Offload ~20 layers to fit in 4GB VRAM
                        verbose=False
                    )
                    logger.info("LLM loaded successfully.")
                except Exception as e:
                    logger.error(f"Failed to load Llama model: {e}")
                    raise
            return _llama_instance

    def is_available(self) -> bool:
        """Check if the embedded model is available (file exists)."""
        return os.path.exists(self.model_path)

    def get_status(self) -> dict:
        """Get LLM availability status."""
        available = self.is_available()
        return {
            "available": available,
            "backend": "llama.cpp (Embedded)",
            "message": "LLM Ready" if available else f"Model not found at {self.model_path}. Run setup.bat.",
        }

    def _save_cache(self):
        """Save cache to disk."""
        try:
            with open(LLM_CACHE_PATH, "w", encoding="utf-8") as f:
                json.dump(_LLM_CACHE, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save LLM cache: {e}")

    def generate(self, prompt: str, system_prompt: str = None, cache_key: str = None, temperature: float = 0.1) -> str:
        """Generate text from the embedded LLM."""
        # 1. Check Cache
        if cache_key and cache_key in _LLM_CACHE:
            logger.info(f"LLM Cache hit for: {cache_key}")
            return _LLM_CACHE[cache_key]

        if not self.is_available():
            raise RuntimeError("Embedded LLM is not available (model file missing).")

        llama = self._get_instance()
        
        # 2. Build Messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        logger.info("Sending request to embedded LLM...")
        
        # 3. Generate
        try:
            with _llama_lock:
                response = llama.create_chat_completion(
                    messages=messages,
                    temperature=temperature,
                    max_tokens=1500,
                )
            
            content = response["choices"][0]["message"]["content"]
            
            # 4. Save Cache
            if cache_key:
                _LLM_CACHE[cache_key] = content
                self._save_cache()
                
            return content
            
        except Exception as e:
            logger.error(f"Embedded LLM error: {e}")
            raise RuntimeError(f"Embedded LLM error: {e}")

    def generate_json(self, prompt: str, system_prompt: str = None, cache_key: str = None, temperature: float = 0.1) -> dict:
        """
        Generate text and parse as JSON.
        We append explicit instructions to return ONLY JSON.
        """
        # Force JSON instruction for the model
        json_sys_prompt = "You are a data extraction AI. You MUST output ONLY raw JSON. No markdown formatting, no backticks, no conversational text."
        if system_prompt:
            json_sys_prompt = f"{system_prompt}\n\n{json_sys_prompt}"
            
        raw_output = self.generate(
            prompt=prompt,
            system_prompt=json_sys_prompt,
            cache_key=cache_key,
            temperature=temperature
        )
        
        return self._extract_json(raw_output)

    def _extract_json(self, text: str) -> dict:
        """Extract JSON from potentially messy LLM output."""
        text = text.strip()
        
        # 1. Direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
            
        # 2. Extract from markdown code blocks
        match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
                
        # 3. Find first { and last }
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start:end+1])
            except json.JSONDecodeError:
                pass
                
        logger.error(f"Failed to extract JSON from LLM output. Raw: {text[:100]}...")
        return {"error": "Failed to parse JSON response"}
