from __future__ import annotations

from local_rag.config import LocalRagConfig
from local_rag.hybrid_search import HybridSearcher
from local_rag.models import ChunkRecord, RankedPaperResult, SearchResponse


class LocalRetrievalEngine:
    def __init__(self, config: LocalRagConfig | None = None):
        self.config = config or LocalRagConfig.from_env()
        self.searcher = HybridSearcher(self.config)

    def search(self, query: str, *, top_k: int | None = None) -> SearchResponse:
        payload = self.searcher.search(query, top_k=top_k)
        chunks = [ChunkRecord(**chunk) for chunk in payload.get("chunks", [])]
        papers = []
        for paper in payload.get("papers", []):
            top_chunks = [ChunkRecord(**chunk) for chunk in paper.pop("top_chunks", [])]
            papers.append(RankedPaperResult(**paper, top_chunks=top_chunks))

        return SearchResponse(
            query=payload.get("query", query),
            chunks=chunks,
            papers=papers,
        )
