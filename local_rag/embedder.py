from __future__ import annotations

from functools import lru_cache

from sentence_transformers import SentenceTransformer

from local_rag.config import LocalRagConfig


@lru_cache(maxsize=2)
def _load_model(model_name: str) -> SentenceTransformer:
    return SentenceTransformer(model_name)


class EmbeddingModel:
    def __init__(self, config: LocalRagConfig):
        self.config = config
        self.model = _load_model(config.embedding_model)

    def embed_documents(self, texts: list[str], batch_size: int = 16) -> list[list[float]]:
        vectors = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 32,
        )
        return vectors.tolist()

    def embed_query(self, query: str) -> list[float]:
        vector = self.model.encode(
            query,
            normalize_embeddings=True,
        )
        return vector.tolist()
