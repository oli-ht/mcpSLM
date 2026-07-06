import json
from typing import Any

from mcp.server.fastmcp import FastMCP

from local_rag.ingest import index_text_folder
from local_rag.search import LocalRetrievalEngine

mcp = FastMCP("local-paper-retrieval")


@mcp.tool()
def index_papers(text_dir: str, reset: bool = False) -> str:
    """Index all .txt papers in a folder: chunk, embed, store in Qdrant + BM25."""
    result = index_text_folder(text_dir, reset=reset)
    return json.dumps(result, indent=2)


@mcp.tool()
def search_papers(
    query: str,
    top_k: int = 20,
) -> str:
    """Hybrid search over indexed papers with dense + BM25 + reranking."""
    engine = LocalRetrievalEngine()
    response = engine.search(query, top_k=top_k)
    payload: dict[str, Any] = {
        "query": response.query,
        "chunks": [chunk.__dict__ for chunk in response.chunks],
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
            for paper in response.papers
        ],
    }
    return json.dumps(payload, indent=2)


if __name__ == "__main__":
    mcp.run(transport="stdio")
