from __future__ import annotations

import json
from pathlib import Path

from local_rag.bm25_index import BM25Index
from local_rag.chunker import chunk_document
from local_rag.config import LocalRagConfig
from local_rag.embedder import EmbeddingModel
from local_rag.models import ChunkRecord
from local_rag.pdf_loader import (
    discover_text_files,
    load_optional_metadata,
    lookup_metadata,
    read_text_file,
)
from local_rag.qdrant_store import QdrantStore


def index_text_folder(
    text_dir: str | Path,
    *,
    metadata_xlsx: str | Path | None = None,
    reset: bool = False,
    config: LocalRagConfig | None = None,
) -> dict:
    cfg = config or LocalRagConfig.from_env()
    text_root = Path(text_dir).expanduser().resolve()
    if not text_root.exists():
        raise FileNotFoundError(f"Text folder not found: {text_root}")

    xlsx_path = Path(metadata_xlsx).expanduser().resolve() if metadata_xlsx else None
    optional_metadata = load_optional_metadata(text_root, metadata_xlsx=xlsx_path)
    text_paths = discover_text_files(text_root)
    if not text_paths:
        raise ValueError(f"No .txt files found under {text_root}")

    embedder = EmbeddingModel(cfg)
    qdrant = QdrantStore(cfg)
    all_chunks: list[ChunkRecord] = []
    indexed_files: list[dict] = []

    for text_path in text_paths:
        text = read_text_file(text_path)
        if not text.strip():
            continue

        metadata = lookup_metadata(optional_metadata, text_path)
        chunks = chunk_document(text_path, text_root, text, metadata, cfg)
        all_chunks.extend(chunks)
        indexed_files.append(
            {
                "file_name": text_path.name,
                "document_id": chunks[0].document_id if chunks else None,
                "chunk_count": len(chunks),
            }
        )

    if not all_chunks:
        raise ValueError("No text could be read from the provided .txt files.")

    sample_vector = embedder.embed_query("dimension probe")
    if reset:
        qdrant.reset_collection(len(sample_vector))
    else:
        qdrant.ensure_collection(len(sample_vector))

    vectors = embedder.embed_documents([chunk.text for chunk in all_chunks])
    qdrant.upsert_chunks(all_chunks, vectors)

    bm25 = BM25Index(cfg)
    bm25.build(all_chunks)
    bm25.save()

    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "text_dir": str(text_root),
        "metadata_xlsx": str(xlsx_path) if xlsx_path else None,
        "file_count": len(indexed_files),
        "chunk_count": len(all_chunks),
        "files": indexed_files,
    }
    with cfg.manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    return manifest


def index_pdf_folder(
    pdf_dir: str | Path,
    *,
    metadata_xlsx: str | Path | None = None,
    reset: bool = False,
    config: LocalRagConfig | None = None,
) -> dict:
    """Backward-compatible alias; indexes a folder of cleaned .txt files."""
    return index_text_folder(
        pdf_dir,
        metadata_xlsx=metadata_xlsx,
        reset=reset,
        config=config,
    )
