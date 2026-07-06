from dataclasses import dataclass, field
from typing import Any


@dataclass
class ChunkRecord:
    chunk_id: str
    document_id: str
    file_name: str
    text: str
    page_number: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    dense_score: float = 0.0
    bm25_score: float = 0.0
    hybrid_score: float = 0.0
    rerank_score: float = 0.0
    final_score: float = 0.0


@dataclass
class RankedPaperResult:
    document_id: str
    title: str
    file_name: str
    final_score: float
    hybrid_score: float
    rerank_score: float
    dense_score: float
    bm25_score: float
    top_chunks: list[ChunkRecord] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchResponse:
    query: str
    chunks: list[ChunkRecord]
    papers: list[RankedPaperResult]
