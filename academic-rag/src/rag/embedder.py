import re
from typing import List, Union

import numpy as np
from loguru import logger
from sentence_transformers import SentenceTransformer

from src.storage.sqlite_cache import SQLiteTTLCache
from src.utils.cache import TTLCache, cache_key


class Embedder:
    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        device: str = "cpu",
        batch_size: int = 32,
        query_cache_enabled: bool = True,
        query_cache_max_size: int = 10000,
        query_cache_ttl_seconds: int = 1800,
        cache_db_path: str | None = None,
    ):
        logger.info("Loading embedding model: {}", model_name)
        self._model_name = model_name
        self._device = device
        self.model = SentenceTransformer(model_name, device=device)
        self.batch_size = batch_size
        get_dim = getattr(self.model, "get_embedding_dimension", None) or self.model.get_sentence_embedding_dimension
        self.dimension = get_dim()

        self.query_cache_enabled = query_cache_enabled
        self.query_cache = None
        if query_cache_enabled:
            if cache_db_path:
                self.query_cache = SQLiteTTLCache[str, np.ndarray](
                    db_path=cache_db_path,
                    namespace=f"embedding:{cache_key(model_name)[:16]}",
                    max_size=query_cache_max_size,
                    ttl_seconds=query_cache_ttl_seconds,
                    value_codec="ndarray",
                )
            else:
                self.query_cache = TTLCache[str, np.ndarray](
                    max_size=query_cache_max_size,
                    ttl_seconds=query_cache_ttl_seconds,
                )

        logger.info("Embedding dimension: {}", self.dimension)

    def embed(self, texts: Union[str, List[str]], normalize: bool = True) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]

        try:
            return self.model.encode(
                texts,
                batch_size=self.batch_size,
                normalize_embeddings=normalize,
                show_progress_bar=len(texts) > 100,
            )
        except RuntimeError as exc:
            if "CUDA" in str(exc) and self._device != "cpu":
                logger.warning("CUDA error during embedding, reinitializing on CPU: {}", exc)
                self._device = "cpu"
                self.model = SentenceTransformer(self._model_name, device="cpu")
                return self.model.encode(
                    texts,
                    batch_size=self.batch_size,
                    normalize_embeddings=normalize,
                    show_progress_bar=len(texts) > 100,
                )
            raise

    def _format_query_for_embedding(self, query: str) -> str:
        q = query.strip()
        if not q:
            return q

        cjk_count = len(re.findall(r"[\u4e00-\u9fff]", q))
        cjk_ratio = cjk_count / max(len(q), 1)
        if cjk_ratio >= 0.15:
            instruction = "为这个句子生成表示以用于检索相关文章："
        else:
            instruction = "Represent this sentence for searching relevant passages: "
        return instruction + q

    def _cache_key(self, query_text: str) -> str:
        return f"vec:{cache_key(f'{self._model_name}|{query_text}')}"

    def embed_query(self, query: str) -> np.ndarray:
        text = self._format_query_for_embedding(query)

        if not self.query_cache_enabled or self.query_cache is None:
            return self.embed(text)

        cache_key = self._cache_key(text)
        cached = self.query_cache.get(cache_key)
        if cached is not None:
            return cached.copy()

        result = self.embed(text)
        self.query_cache.set(cache_key, result.copy())
        return result

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        return self.embed(texts)
