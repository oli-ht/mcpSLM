from __future__ import annotations

import hashlib
from pathlib import Path

from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document as LlamaDocument

from local_rag.config import LocalRagConfig
from local_rag.models import ChunkRecord


def _document_id(file_path: Path, source_dir: Path) -> str:
    relative = file_path.relative_to(source_dir)
    stem_key = relative.with_suffix("").as_posix()
    return hashlib.sha1(stem_key.encode("utf-8")).hexdigest()[:16]


def _chunk_id(document_id: str, chunk_index: int) -> str:
    return f"{document_id}:{chunk_index}"


def chunk_document(
    file_path: Path,
    source_dir: Path,
    text: str,
    metadata: dict,
    config: LocalRagConfig,
) -> list[ChunkRecord]:
    document_id = _document_id(file_path, source_dir)
    splitter = SentenceSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
    )
    llama_doc = LlamaDocument(
        text=text,
        metadata={
            "document_id": document_id,
            "file_name": file_path.name,
        },
    )
    nodes = splitter.get_nodes_from_documents([llama_doc])

    chunks: list[ChunkRecord] = []
    for index, node in enumerate(nodes):
        chunks.append(
            ChunkRecord(
                chunk_id=_chunk_id(document_id, index),
                document_id=document_id,
                file_name=file_path.name,
                text=node.get_content(),
                page_number=node.metadata.get("page_label"),
                metadata={
                    **metadata,
                    "source_path": str(file_path),
                },
            )
        )
    return chunks


def chunk_pdf(
    pdf_path: Path,
    pdf_dir: Path,
    text: str,
    metadata: dict,
    config: LocalRagConfig,
) -> list[ChunkRecord]:
    return chunk_document(pdf_path, pdf_dir, text, metadata, config)
