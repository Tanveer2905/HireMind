import json
import logging
import re
import os
import threading

os.environ["GGML_NO_AVX2"] = "1"
os.environ["GGML_NO_FMA"] = "1"
os.environ["GGML_NO_F16C"] = "1"

from utils import LLAMA3_MODEL_PATH
from backend.user_context import get_user_dir

logger = logging.getLogger(__name__)

_llama_instance = None
_llama_lock = threading.Lock()

class LlamaEngine:
    def __init__(self, user_id: str, model_path=str(LLAMA3_MODEL_PATH)):
        self.user_id = user_id
        self.model_path = model_path
        self.cache_path = os.path.join(get_user_dir(user_id), "llm_cache.json")
        self._cache = self._load_cache()

    def _load_cache(self):
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def _save_cache(self):
        try:
            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump(self._cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save LLM cache: {e}")

    def _get_instance(self):
        global _llama_instance
        with _llama_lock:
            if _llama_instance is None:
                if not os.path.exists(self.model_path):
                    raise FileNotFoundError(f"LLM Model not found at {self.model_path}")
                logger.info(f"Loading local LLM into memory: {self.model_path}")
                from llama_cpp import Llama
                _llama_instance = Llama(
                    model_path=self.model_path,
                    n_ctx=4096,
                    n_gpu_layers=20,
                    verbose=False
                )
                logger.info("LLM loaded successfully.")
            return _llama_instance

    def is_available(self) -> bool:
        return os.path.exists(self.model_path)

    def generate(self, prompt: str, system_prompt: str = None, cache_key: str = None, temperature: float = 0.1) -> str:
        if cache_key and cache_key in self._cache:
            return self._cache[cache_key]

        if not self.is_available():
            raise RuntimeError("Embedded LLM is not available.")

        llama = self._get_instance()
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            with _llama_lock:
                response = llama.create_chat_completion(
                    messages=messages,
                    temperature=temperature,
                    max_tokens=1500,
                )
            content = response["choices"][0]["message"]["content"]
            if cache_key:
                self._cache[cache_key] = content
                self._save_cache()
            return content
        except Exception as e:
            raise RuntimeError(f"Embedded LLM error: {e}")

    def generate_json(self, prompt: str, system_prompt: str = None, cache_key: str = None, temperature: float = 0.1) -> dict:
        json_sys_prompt = "You are a data extraction AI. You MUST output ONLY raw JSON. No markdown formatting, no backticks, no conversational text."
        if system_prompt:
            json_sys_prompt = f"{system_prompt}\n\n{json_sys_prompt}"
            
        raw_output = self.generate(prompt=prompt, system_prompt=json_sys_prompt, cache_key=cache_key, temperature=temperature)
        return self._extract_json(raw_output)

    def _extract_json(self, text: str) -> dict:
        text = text.strip()
        try: return json.loads(text)
        except json.JSONDecodeError: pass
            
        match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
        if match:
            try: return json.loads(match.group(1))
            except json.JSONDecodeError: pass
                
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try: return json.loads(text[start:end+1])
            except json.JSONDecodeError: pass
                
        return {"error": "Failed to parse JSON response"}
