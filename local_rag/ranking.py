from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import Any

from local_rag.models import ChunkRecord, RankedPaperResult


def _metadata_value(metadata: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in metadata and metadata[key] is not None:
            return metadata[key]
    return None


def recency_weight(metadata: dict[str, Any], current_year: int | None = None) -> float:
    year = _metadata_value(metadata, "year", "publication_year", "date")
    if year is None:
        return 0.5

    if isinstance(year, str):
        digits = "".join(ch for ch in year if ch.isdigit())
        if len(digits) >= 4:
            year = int(digits[:4])
        else:
            return 0.5

    current = current_year or datetime.now().year
    age = max(0, current - int(year))
    return max(0.1, 1.0 - (age / 20.0))


def aggregate_papers(
    chunks: list[ChunkRecord],
    *,
    hybrid_weight: float = 0.45,
    rerank_weight: float = 0.35,
    recency_weight_factor: float = 0.1,
    authority_weight_factor: float = 0.1,
) -> list[RankedPaperResult]:
    grouped: dict[str, list[ChunkRecord]] = defaultdict(list)
    for chunk in chunks:
        grouped[chunk.document_id].append(chunk)

    ranked: list[RankedPaperResult] = []
    for document_id, doc_chunks in grouped.items():
        doc_chunks.sort(key=lambda item: item.final_score, reverse=True)
        best = doc_chunks[0]
        metadata = dict(best.metadata)
        recency = recency_weight(metadata)
        authority = 0.7 if _metadata_value(metadata, "venue", "journal", "conference") else 0.5

        final_score = (
            hybrid_weight * best.hybrid_score
            + rerank_weight * best.rerank_score
            + recency_weight_factor * recency
            + authority_weight_factor * authority
        )

        title = (
            _metadata_value(metadata, "title")
            or best.file_name
            or document_id
        )

        ranked.append(
            RankedPaperResult(
                document_id=document_id,
                title=str(title),
                file_name=best.file_name,
                final_score=round(final_score, 4),
                hybrid_score=round(best.hybrid_score, 4),
                rerank_score=round(best.rerank_score, 4),
                dense_score=round(best.dense_score, 4),
                bm25_score=round(best.bm25_score, 4),
                top_chunks=doc_chunks[:5],
                metadata=metadata,
            )
        )

    ranked.sort(key=lambda paper: paper.final_score, reverse=True)
    return ranked
