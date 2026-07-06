#!/usr/bin/env python3
"""CLI for the local embedding + hybrid retrieval pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from local_rag.ingest import index_text_folder
from local_rag.search import LocalRetrievalEngine


def run_index(args: argparse.Namespace) -> None:
    result = index_text_folder(
        args.text_dir,
        metadata_xlsx=args.metadata,
        reset=args.reset,
    )
    print(json.dumps(result, indent=2))


def run_search(args: argparse.Namespace) -> None:
    engine = LocalRetrievalEngine()
    response = engine.search(args.query, top_k=args.top_k)

    print(f"Query: {response.query}\n")
    if not response.papers:
        print("No ranked papers returned.")
        return

    print("Ranked papers:")
    for index, paper in enumerate(response.papers, start=1):
        print(
            f"{index}. {paper.title} "
            f"(final={paper.final_score:.3f}, hybrid={paper.hybrid_score:.3f}, "
            f"rerank={paper.rerank_score:.3f}, dense={paper.dense_score:.3f}, "
            f"bm25={paper.bm25_score:.3f})"
        )
        for chunk in paper.top_chunks[:2]:
            excerpt = chunk.text.replace("\n", " ")[:160]
            print(
                f"   - final={chunk.final_score:.3f}, hybrid={chunk.hybrid_score:.3f}: "
                f"{excerpt}..."
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Local text embedding pipeline with Qdrant + BM25 hybrid search"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    index_parser = subparsers.add_parser("index", help="Index a folder of cleaned .txt files")
    index_parser.add_argument("text_dir", help="Folder containing .txt paper files")
    index_parser.add_argument(
        "--metadata",
        help="Path to sample_metadata.xlsx (or similar) with a PDF_file column",
    )
    index_parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete and rebuild the Qdrant collection",
    )

    search_parser = subparsers.add_parser("search", help="Search indexed papers")
    search_parser.add_argument("query", help="Natural language query")
    search_parser.add_argument("--top-k", type=int, default=20, help="Final reranked results")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "index":
        run_index(args)
    elif args.command == "search":
        run_search(args)


if __name__ == "__main__":
    main()
