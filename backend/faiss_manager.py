import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from backend.user_context import get_faiss_dir

logger = logging.getLogger(__name__)

BGE_MODEL_PATH = Path(os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "bge-small-en-v1.5"))
BGE_BASE_MODEL_PATH = Path(os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "bge-base-en-v1.5"))

class UserFAISSCache:
    def __init__(self, user_id: str, embedding_dim: int):
        self.user_id = user_id
        self.cache_path = os.path.join(get_faiss_dir(user_id), "embedding_cache.npz")
        self.embedding_dim = embedding_dim
        self._cache: Dict[str, np.ndarray] = {}
        self._hash_cache: Dict[str, str] = {}
        self.index: faiss.IndexFlatIP | None = None
        self._filenames: List[str] = []

    def load_cache(self) -> None:
        self._cache = {}
        self._hash_cache = {}
        if os.path.exists(self.cache_path):
            try:
                data = np.load(self.cache_path, allow_pickle=True)
                if "keys" in data and "values" in data:
                    for key, val in zip(data["keys"], data["values"]):
                        self._cache[str(key)] = val
                if "hash_keys" in data and "hash_values" in data:
                    for hk, hv in zip(data["hash_keys"], data["hash_values"]):
                        self._hash_cache[str(hk)] = str(hv)
            except Exception as e:
                logger.warning(f"Failed to load user cache: {e}")

    def save_cache(self) -> None:
        if not self._cache:
            return
        emb_keys = list(self._cache.keys())
        emb_values = np.stack([self._cache[k] for k in emb_keys]).astype(np.float32)
        hash_keys = list(self._hash_cache.keys())
        hash_values = [self._hash_cache[k] for k in hash_keys]
        np.savez(
            self.cache_path,
            keys=np.array(emb_keys, dtype=object),
            values=emb_values,
            hash_keys=np.array(hash_keys, dtype=object),
            hash_values=np.array(hash_values, dtype=object),
        )

class FaissManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(FaissManager, cls).__new__(cls)
            cls._instance.model = None
            cls._instance.embedding_dim = 768
            cls._instance.user_caches: Dict[str, UserFAISSCache] = {}
        return cls._instance

    def load_model(self) -> None:
        if self.model is not None:
            return
        model_path = BGE_BASE_MODEL_PATH if BGE_BASE_MODEL_PATH.exists() else BGE_MODEL_PATH
        logger.info(f"Loading embedding model from {model_path}...")
        self.model = SentenceTransformer(str(model_path))
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

    def get_user_cache(self, user_id: str) -> UserFAISSCache:
        if user_id not in self.user_caches:
            self.user_caches[user_id] = UserFAISSCache(user_id, self.embedding_dim)
        return self.user_caches[user_id]

    def encode(self, texts: List[str], show_progress: bool = False) -> np.ndarray:
        if self.model is None:
            self.load_model()
        return self.model.encode(texts, show_progress_bar=show_progress, normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)

    def encode_query(self, query: str) -> np.ndarray:
        prefixed = f"Represent this sentence for searching relevant passages: {query}"
        return self.encode([prefixed])[0]

    def get_or_compute_embeddings(self, user_id: str, parsed_resumes: List[dict], file_hashes: Dict[str, str]) -> Tuple[np.ndarray, List[str]]:
        if self.model is None:
            self.load_model()
        cache = self.get_user_cache(user_id)
        cache.load_cache()

        all_embeddings, all_filenames, to_encode, to_encode_indices = [], [], [], []

        for i, resume in enumerate(parsed_resumes):
            filename = resume["filename"]
            if filename in cache._cache and filename in cache._hash_cache:
                if str(cache._hash_cache[filename]) == str(file_hashes.get(filename, "")):
                    all_embeddings.append(cache._cache[filename])
                    all_filenames.append(filename)
                    continue
            
            to_encode.append(resume.get("raw_text", ""))
            to_encode_indices.append(i)
            all_filenames.append(filename)

        if to_encode:
            new_embeddings = self.encode(to_encode, show_progress=True)
            embed_idx = 0
            final_embeddings = []
            for i, resume in enumerate(parsed_resumes):
                filename = resume["filename"]
                if i in to_encode_indices:
                    emb = new_embeddings[embed_idx]
                    embed_idx += 1
                    cache._cache[filename] = emb
                    if filename in file_hashes:
                        cache._hash_cache[filename] = file_hashes[filename]
                    final_embeddings.append(emb)
                else:
                    final_embeddings.append(cache._cache.get(filename, np.zeros(self.embedding_dim)))
            all_embeddings = final_embeddings
            cache.save_cache()

        return np.stack(all_embeddings).astype(np.float32), list(all_filenames)

    def build_index(self, user_id: str, embeddings: np.ndarray, filenames: List[str]) -> None:
        cache = self.get_user_cache(user_id)
        n, dim = embeddings.shape
        cache._filenames = list(filenames)
        cache.index = faiss.IndexFlatIP(dim)
        cache.index.add(embeddings)

    def search(self, user_id: str, query_embedding: np.ndarray, top_k: int = 50) -> List[Dict[str, Any]]:
        cache = self.get_user_cache(user_id)
        if cache.index is None or cache.index.ntotal == 0:
            return []
        
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        k = min(top_k, cache.index.ntotal)
        scores, indices = cache.index.search(query_embedding.astype(np.float32), k)
        
        results = []
        for i in range(k):
            idx = int(indices[0][i])
            if idx < 0: continue
            results.append({
                "filename": cache._filenames[idx] if idx < len(cache._filenames) else f"unknown_{idx}",
                "score": float(scores[0][i]),
                "index": idx
            })
        return results
