"""
embeddings.py — Embedding Engine and FAISS Vector Indexing
Loads BAAI/bge-small-en from local /models directory.
Generates, caches, and indexes embeddings for fast similarity search.
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from utils import (
    BGE_MODEL_PATH, BGE_BASE_MODEL_PATH, EMBEDDING_CACHE_PATH, DATA_DIR,
    file_hash,
)

logger = logging.getLogger(__name__)


class EmbeddingEngine:
    """
    Manages embedding generation and FAISS indexing.
    - Loads model exclusively from local /models directory
    - Caches embeddings to disk
    - Supports incremental updates (only embeds new/changed documents)
    """

    def __init__(self, model_path: Path | None = None):
        # Prefer bge-base-en-v1.5, fall back to bge-small-en
        if model_path:
            self.model_path = model_path
        elif BGE_BASE_MODEL_PATH.exists() and any(BGE_BASE_MODEL_PATH.iterdir()):
            self.model_path = BGE_BASE_MODEL_PATH
            logger.info("Using bge-base-en-v1.5 embedding model")
        else:
            self.model_path = BGE_MODEL_PATH
            logger.info("bge-base-en-v1.5 not found, using bge-small-en fallback")
        self.model: SentenceTransformer | None = None
        self.index: faiss.IndexFlatIP | None = None
        self.embedding_dim: int = 768  # bge-base-en-v1.5 dimension (384 for small)
        self._cache: dict[str, np.ndarray] = {}
        self._filenames: list[str] = []

    def load_model(self) -> None:
        """Load the sentence-transformer model from local path."""
        model_path_str = str(self.model_path)
        if not self.model_path.exists():
            raise RuntimeError(
                f"Embedding model not found at {model_path_str}. "
                "Run setup.bat to download models."
            )
        logger.info(f"Loading embedding model from {model_path_str}...")
        self.model = SentenceTransformer(model_path_str)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info(
            f"Model loaded. Embedding dimension: {self.embedding_dim}"
        )

    def encode(
        self,
        texts: list[str],
        batch_size: int = 32,
        show_progress: bool = True,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Encode texts into dense vectors.
        
        Args:
            texts: List of text strings to encode
            batch_size: Batch size for encoding
            show_progress: Show progress bar
            normalize: L2-normalize embeddings (required for cosine sim via inner product)
        
        Returns:
            np.ndarray of shape (len(texts), embedding_dim)
        """
        if self.model is None:
            self.load_model()

        # BGE models benefit from a query prefix for retrieval
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=normalize,
            convert_to_numpy=True,
        )

        return embeddings.astype(np.float32)

    def encode_query(self, query: str) -> np.ndarray:
        """
        Encode a query (job description) with the instruction prefix
        recommended for BGE models.
        """
        # BGE models use an instruction prefix for queries
        prefixed = f"Represent this sentence for searching relevant passages: {query}"
        embedding = self.encode([prefixed], show_progress=False, normalize=True)
        return embedding[0]

    def build_index(
        self,
        embeddings: np.ndarray,
        filenames: list[str],
    ) -> None:
        """
        Build a FAISS IndexFlatIP (inner product) index.
        Using IP on L2-normalized vectors is equivalent to cosine similarity.
        
        Args:
            embeddings: np.ndarray of shape (n, dim)
            filenames: List of corresponding filenames for lookup
        """
        n, dim = embeddings.shape
        self.embedding_dim = dim
        self._filenames = list(filenames)

        # Inner product index (cosine similarity on normalized vectors)
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)

        logger.info(f"FAISS index built with {n} vectors (dim={dim})")

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 100,
    ) -> list[dict[str, Any]]:
        """
        Search the FAISS index for top-K most similar candidates.
        
        Args:
            query_embedding: 1D array of shape (dim,)
            top_k: Number of results to return
        
        Returns:
            List of dicts with 'filename', 'score', 'index'
        """
        if self.index is None or self.index.ntotal == 0:
            logger.warning("FAISS index is empty or not built")
            return []

        # Ensure correct shape
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query_embedding.astype(np.float32), k)

        results = []
        for i in range(k):
            idx = int(indices[0][i])
            if idx < 0:
                continue  # FAISS returns -1 for missing entries
            results.append({
                "filename": self._filenames[idx] if idx < len(self._filenames) else f"unknown_{idx}",
                "score": float(scores[0][i]),
                "index": idx,
            })

        return results

    # ------------------------------------------------------------------
    # Embedding cache management
    # ------------------------------------------------------------------
    def load_cache(self) -> dict[str, np.ndarray]:
        """Load cached embeddings and hashes from disk."""
        self._cache = {}
        self._hash_cache = {}

        if EMBEDDING_CACHE_PATH.exists():
            try:
                data = np.load(str(EMBEDDING_CACHE_PATH), allow_pickle=True)
                if "keys" in data and "values" in data:
                    keys = data["keys"]
                    values = data["values"]
                    for key, val in zip(keys, values):
                        self._cache[str(key)] = val
                # Load hashes separately
                if "hash_keys" in data and "hash_values" in data:
                    hkeys = data["hash_keys"]
                    hvals = data["hash_values"]
                    for hk, hv in zip(hkeys, hvals):
                        self._hash_cache[str(hk)] = str(hv)
                logger.info(f"Loaded {len(self._cache)} cached embeddings")
            except Exception as e:
                logger.warning(f"Failed to load embedding cache: {e}")

        return self._cache

    def save_cache(self) -> None:
        """Save embedding cache and hashes to disk."""
        if not self._cache:
            return

        EMBEDDING_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)

        # Embeddings: all same shape, can be stacked into a uniform array
        emb_keys = list(self._cache.keys())
        emb_values = np.stack([self._cache[k] for k in emb_keys]).astype(np.float32)

        # Hashes: string arrays stored separately
        hash_keys = list(self._hash_cache.keys())
        hash_values = [self._hash_cache[k] for k in hash_keys]

        np.savez(
            str(EMBEDDING_CACHE_PATH),
            keys=np.array(emb_keys, dtype=object),
            values=emb_values,
            hash_keys=np.array(hash_keys, dtype=object),
            hash_values=np.array(hash_values, dtype=object),
        )
        logger.info(f"Saved {len(emb_keys)} embeddings to cache")

    def get_or_compute_embeddings(
        self,
        parsed_resumes: list[dict],
        file_hashes: dict[str, str] | None = None,
    ) -> tuple[np.ndarray, list[str]]:
        """
        Get embeddings for parsed resumes, using cache where possible.
        Only computes embeddings for new/changed resumes.
        
        Args:
            parsed_resumes: List of parsed resume dicts (must have 'filename' and 'raw_text')
            file_hashes: Dict of filename → hash for cache validation
        
        Returns:
            Tuple of (embeddings_array, filenames_list)
        """
        if self.model is None:
            self.load_model()

        # Load existing cache
        self.load_cache()

        all_embeddings = []
        all_filenames = []
        to_encode = []
        to_encode_indices = []

        for i, resume in enumerate(parsed_resumes):
            filename = resume["filename"]

            # Check if we have a valid cached embedding
            if (
                file_hashes
                and filename in self._cache
                and filename in self._hash_cache
            ):
                cached_hash = self._hash_cache[filename]
                current_hash = file_hashes.get(filename, "")
                if str(cached_hash) == str(current_hash):
                    all_embeddings.append(self._cache[filename])
                    all_filenames.append(filename)
                    continue

            # Need to compute embedding
            to_encode.append(resume.get("raw_text", ""))
            to_encode_indices.append(i)
            all_filenames.append(filename)

        # Batch encode new texts
        if to_encode:
            logger.info(f"Computing embeddings for {len(to_encode)} resumes...")
            new_embeddings = self.encode(to_encode, show_progress=True)

            # Insert into correct positions and update cache
            embed_idx = 0
            final_embeddings = []

            for i, resume in enumerate(parsed_resumes):
                filename = resume["filename"]
                if i in to_encode_indices:
                    emb = new_embeddings[embed_idx]
                    embed_idx += 1
                    # Update cache
                    self._cache[filename] = emb
                    if file_hashes and filename in file_hashes:
                        self._hash_cache[filename] = file_hashes[filename]
                    final_embeddings.append(emb)
                else:
                    final_embeddings.append(
                        self._cache.get(filename, np.zeros(self.embedding_dim))
                    )

            all_embeddings = final_embeddings
            self.save_cache()
        else:
            logger.info("All embeddings loaded from cache")

        embeddings_array = np.stack(all_embeddings).astype(np.float32)
        return embeddings_array, all_filenames

