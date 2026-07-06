from __future__ import annotations

from functools import lru_cache

from sentence_transformers import CrossEncoder

from local_rag.config import LocalRagConfig
from local_rag.models import ChunkRecord


@lru_cache(maxsize=2)
def _load_cross_encoder(model_name: str) -> CrossEncoder:
    return CrossEncoder(model_name)


class Reranker:
    def __init__(self, config: LocalRagConfig):
        self.config = config
        self.cross_encoder = _load_cross_encoder(config.reranker_model)
        self.cohere_client = None
        if config.cohere_api_key:
            try:
                import cohere

                self.cohere_client = cohere.ClientV2(api_key=config.cohere_api_key)
            except ImportError:
                self.cohere_client = None

    def rerank(self, query: str, chunks: list[ChunkRecord], top_k: int) -> list[ChunkRecord]:
        if not chunks:
            return []

        if self.cohere_client is not None:
            return self._rerank_cohere(query, chunks, top_k)
        return self._rerank_bge(query, chunks, top_k)

    def _rerank_bge(self, query: str, chunks: list[ChunkRecord], top_k: int) -> list[ChunkRecord]:
        pairs = [(query, chunk.text) for chunk in chunks]
        scores = self.cross_encoder.predict(pairs)
        ranked = sorted(
            zip(chunks, scores),
            key=lambda item: float(item[1]),
            reverse=True,
        )[:top_k]

        results: list[ChunkRecord] = []
        max_score = max(float(score) for _, score in ranked) if ranked else 1.0
        for chunk, score in ranked:
            normalized = float(score) / max_score if max_score > 0 else 0.0
            chunk.rerank_score = round(normalized, 4)
            chunk.final_score = round(
                0.6 * chunk.hybrid_score + 0.4 * chunk.rerank_score,
                4,
            )
            results.append(chunk)
        return results

    def _rerank_cohere(self, query: str, chunks: list[ChunkRecord], top_k: int) -> list[ChunkRecord]:
        response = self.cohere_client.rerank(
            model="rerank-english-v3.0",
            query=query,
            documents=[chunk.text for chunk in chunks],
            top_n=min(top_k, len(chunks)),
        )

        ranked: list[ChunkRecord] = []
        for item in response.results:
            chunk = chunks[item.index]
            chunk.rerank_score = round(float(item.relevance_score), 4)
            chunk.final_score = round(
                0.6 * chunk.hybrid_score + 0.4 * chunk.rerank_score,
                4,
            )
            ranked.append(chunk)
        return ranked
