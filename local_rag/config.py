import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class LocalRagConfig:
    data_dir: Path
    qdrant_path: Path
    collection_name: str
    embedding_model: str
    reranker_model: str
    cohere_api_key: str | None
    chunk_size: int
    chunk_overlap: int
    vector_top_k: int
    bm25_top_k: int
    rerank_top_k: int
    rrf_k: int

    @classmethod
    def from_env(cls) -> "LocalRagConfig":
        data_dir = Path(os.getenv("LOCAL_RAG_DATA_DIR", PROJECT_ROOT / "local_rag_data"))
        return cls(
            data_dir=data_dir,
            qdrant_path=Path(os.getenv("QDRANT_PATH", data_dir / "qdrant")),
            collection_name=os.getenv("QDRANT_COLLECTION", "papers"),
            embedding_model=os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"),
            reranker_model=os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
            cohere_api_key=os.getenv("COHERE_API_KEY") or None,
            chunk_size=int(os.getenv("CHUNK_SIZE", "512")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "64")),
            vector_top_k=int(os.getenv("VECTOR_TOP_K", "50")),
            bm25_top_k=int(os.getenv("BM25_TOP_K", "50")),
            rerank_top_k=int(os.getenv("RERANK_TOP_K", "20")),
            rrf_k=int(os.getenv("RRF_K", "60")),
        )

    @property
    def bm25_index_path(self) -> Path:
        return self.data_dir / "bm25_index.json"

    @property
    def manifest_path(self) -> Path:
        return self.data_dir / "manifest.json"
