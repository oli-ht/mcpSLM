from __future__ import annotations

from collections import defaultdict

from local_rag.bm25_index import BM25Index
from local_rag.config import LocalRagConfig
from local_rag.embedder import EmbeddingModel
from local_rag.models import ChunkRecord
from local_rag.qdrant_store import QdrantStore
from local_rag.ranking import aggregate_papers
from local_rag.reranker import Reranker


def reciprocal_rank_fusion(
    dense_hits: list[tuple[ChunkRecord, float]],
    bm25_hits: list[tuple[ChunkRecord, float]],
    *,
    rrf_k: int = 60,
) -> list[ChunkRecord]:
    fused_scores: dict[str, float] = defaultdict(float)
    chunk_lookup: dict[str, ChunkRecord] = {}

    for ranked in (dense_hits, bm25_hits):
        for rank, (chunk, _) in enumerate(ranked, start=1):
            fused_scores[chunk.chunk_id] += 1.0 / (rrf_k + rank)
            existing = chunk_lookup.get(chunk.chunk_id)
            if existing is None:
                chunk_lookup[chunk.chunk_id] = chunk
            else:
                existing.dense_score = max(existing.dense_score, chunk.dense_score)
                existing.bm25_score = max(existing.bm25_score, chunk.bm25_score)

    merged: list[ChunkRecord] = []
    for chunk_id, score in sorted(fused_scores.items(), key=lambda item: item[1], reverse=True):
        chunk = chunk_lookup[chunk_id]
        chunk.hybrid_score = round(score, 6)
        merged.append(chunk)
    return merged


def _normalize_dense_scores(chunks: list[ChunkRecord]) -> None:
    if not chunks:
        return
    max_score = max(chunk.dense_score for chunk in chunks) or 1.0
    for chunk in chunks:
        chunk.dense_score = round(chunk.dense_score / max_score, 4)


def _normalize_bm25_scores(chunks: list[ChunkRecord]) -> None:
    if not chunks:
        return
    max_score = max(chunk.bm25_score for chunk in chunks) or 1.0
    for chunk in chunks:
        chunk.bm25_score = round(chunk.bm25_score / max_score, 4)


class HybridSearcher:
    def __init__(self, config: LocalRagConfig):
        self.config = config
        self.embedder = EmbeddingModel(config)
        self.qdrant = QdrantStore(config)
        self.bm25 = BM25Index(config)
        self.reranker = Reranker(config)
        self.bm25.load()

    def search(self, query: str, *, top_k: int | None = None) -> dict:
        rerank_top_k = top_k or self.config.rerank_top_k
        query_vector = self.embedder.embed_query(query)

        dense_hits = self.qdrant.dense_search(query_vector, self.config.vector_top_k)
        bm25_hits = self.bm25.search(query, self.config.bm25_top_k)

        dense_chunks = [chunk for chunk, _ in dense_hits]
        bm25_chunks = [chunk for chunk, _ in bm25_hits]
        _normalize_dense_scores(dense_chunks)
        _normalize_bm25_scores(bm25_chunks)

        merged = reciprocal_rank_fusion(
            dense_hits,
            bm25_hits,
            rrf_k=self.config.rrf_k,
        )
        candidate_pool = merged[: max(self.config.vector_top_k, self.config.bm25_top_k)]
        reranked = self.reranker.rerank(query, candidate_pool, rerank_top_k)
        papers = aggregate_papers(reranked)

        return {
            "query": query,
            "chunks": [chunk.__dict__ for chunk in reranked],
            "papers": [
                {
                    "document_id": paper.document_id,
                    "title": paper.title,
                    "file_name": paper.file_name,
                    "final_score": paper.final_score,
                    "hybrid_score": paper.hybrid_score,
                    "rerank_score": paper.rerank_score,
                    "dense_score": paper.dense_score,
                    "bm25_score": paper.bm25_score,
                    "metadata": paper.metadata,
                    "top_chunks": [chunk.__dict__ for chunk in paper.top_chunks],
                }
                for paper in papers
            ],
        }
