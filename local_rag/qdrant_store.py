from __future__ import annotations

import uuid
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from local_rag.config import LocalRagConfig
from local_rag.models import ChunkRecord


class QdrantStore:
    def __init__(self, config: LocalRagConfig):
        self.config = config
        config.qdrant_path.mkdir(parents=True, exist_ok=True)
        self.client = QdrantClient(path=str(config.qdrant_path))

    def ensure_collection(self, vector_size: int) -> None:
        if self.client.collection_exists(self.config.collection_name):
            info = self.client.get_collection(self.config.collection_name)
            current_size = info.config.params.vectors.size
            if current_size != vector_size:
                raise ValueError(
                    f"Collection {self.config.collection_name} expects size {current_size}, "
                    f"but embedding model produces {vector_size}. Delete {self.config.qdrant_path} "
                    "or use a new collection name."
                )
            return

        self.client.create_collection(
            collection_name=self.config.collection_name,
            vectors_config=qmodels.VectorParams(
                size=vector_size,
                distance=qmodels.Distance.COSINE,
            ),
        )

    def reset_collection(self, vector_size: int) -> None:
        if self.client.collection_exists(self.config.collection_name):
            self.client.delete_collection(self.config.collection_name)
        self.ensure_collection(vector_size)

    def upsert_chunks(self, chunks: list[ChunkRecord], vectors: list[list[float]]) -> None:
        points = []
        for chunk, vector in zip(chunks, vectors):
            point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, chunk.chunk_id))
            payload: dict[str, Any] = {
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "file_name": chunk.file_name,
                "text": chunk.text,
                "page_number": chunk.page_number,
                "metadata": chunk.metadata,
            }
            points.append(
                qmodels.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload,
                )
            )

        self.client.upsert(
            collection_name=self.config.collection_name,
            points=points,
        )

    def dense_search(self, query_vector: list[float], limit: int) -> list[tuple[ChunkRecord, float]]:
        if not self.client.collection_exists(self.config.collection_name):
            return []

        response = self.client.query_points(
            collection_name=self.config.collection_name,
            query=query_vector,
            limit=limit,
            with_payload=True,
        )

        results: list[tuple[ChunkRecord, float]] = []
        for hit in response.points:
            payload = hit.payload or {}
            chunk = ChunkRecord(
                chunk_id=str(payload.get("chunk_id", hit.id)),
                document_id=str(payload.get("document_id", "")),
                file_name=str(payload.get("file_name", "")),
                text=str(payload.get("text", "")),
                page_number=payload.get("page_number"),
                metadata=dict(payload.get("metadata") or {}),
                dense_score=float(hit.score or 0.0),
            )
            results.append((chunk, chunk.dense_score))
        return results
