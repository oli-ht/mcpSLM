from __future__ import annotations

import json
import re
from dataclasses import asdict
from pathlib import Path

from rank_bm25 import BM25Okapi

from local_rag.config import LocalRagConfig
from local_rag.models import ChunkRecord


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9_]+", text.lower())


class BM25Index:
    def __init__(self, config: LocalRagConfig):
        self.config = config
        self.chunks: list[ChunkRecord] = []
        self._bm25: BM25Okapi | None = None

    def build(self, chunks: list[ChunkRecord]) -> None:
        self.chunks = chunks
        corpus = [_tokenize(chunk.text) for chunk in chunks]
        self._bm25 = BM25Okapi(corpus)

    def save(self) -> None:
        self.config.data_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "chunks": [
                {
                    "chunk_id": chunk.chunk_id,
                    "document_id": chunk.document_id,
                    "file_name": chunk.file_name,
                    "text": chunk.text,
                    "page_number": chunk.page_number,
                    "metadata": chunk.metadata,
                }
                for chunk in self.chunks
            ]
        }
        with self.config.bm25_index_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    def load(self) -> None:
        if not self.config.bm25_index_path.exists():
            self.chunks = []
            self._bm25 = None
            return

        with self.config.bm25_index_path.open(encoding="utf-8") as handle:
            payload = json.load(handle)

        self.chunks = [
            ChunkRecord(
                chunk_id=item["chunk_id"],
                document_id=item["document_id"],
                file_name=item["file_name"],
                text=item["text"],
                page_number=item.get("page_number"),
                metadata=item.get("metadata") or {},
            )
            for item in payload.get("chunks", [])
        ]
        corpus = [_tokenize(chunk.text) for chunk in self.chunks]
        self._bm25 = BM25Okapi(corpus) if corpus else None

    def search(self, query: str, limit: int) -> list[tuple[ChunkRecord, float]]:
        if not self._bm25 or not self.chunks:
            return []

        scores = self._bm25.get_scores(_tokenize(query))
        ranked = sorted(
            ((index, float(score)) for index, score in enumerate(scores)),
            key=lambda item: item[1],
            reverse=True,
        )[:limit]

        results: list[tuple[ChunkRecord, float]] = []
        for index, score in ranked:
            if score <= 0:
                continue
            chunk = self.chunks[index]
            chunk.bm25_score = score
            results.append((chunk, score))
        return results
