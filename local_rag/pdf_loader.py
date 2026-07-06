from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import fitz


def load_metadata_xlsx(xlsx_path: Path) -> dict[str, dict[str, Any]]:
    """Load paper metadata from an Excel file keyed by PDF filename."""
    import pandas as pd

    dataframe = pd.read_excel(xlsx_path)
    if "PDF_file" not in dataframe.columns:
        raise ValueError("Metadata spreadsheet must include a 'PDF_file' column.")

    metadata_by_file: dict[str, dict[str, Any]] = {}
    for _, row in dataframe.iterrows():
        file_name = str(row["PDF_file"]).strip()
        if not file_name or file_name.lower() == "nan":
            continue

        record: dict[str, Any] = {"file_name": file_name}
        field_map = {
            "title": "TI",
            "year": "PY",
            "abstract": "AB",
            "keywords": "DE",
            "document_type": "DT",
            "authors": "RP",
            "citations": "Z9",
            "doi": "DI",
        }
        for target, source in field_map.items():
            value = row.get(source)
            if value is not None and str(value).strip() and str(value).lower() != "nan":
                if target == "year":
                    try:
                        record[target] = int(float(value))
                    except (TypeError, ValueError):
                        record[target] = value
                elif target == "citations":
                    try:
                        record[target] = int(float(value))
                    except (TypeError, ValueError):
                        record[target] = value
                else:
                    record[target] = str(value).strip()

        metadata_by_file[file_name] = record
        stem = Path(file_name).stem
        if stem not in metadata_by_file:
            metadata_by_file[stem] = record

    return metadata_by_file


def lookup_metadata(metadata_by_file: dict[str, dict[str, Any]], file_path: Path) -> dict[str, Any]:
    """Resolve metadata for a document file (.txt, .pdf, etc.)."""
    candidates = (
        file_path.name,
        file_path.stem,
        f"{file_path.stem}.pdf",
        f"{file_path.stem}.txt",
    )
    for key in candidates:
        record = metadata_by_file.get(key)
        if record:
            return dict(record)
    return {}


def load_optional_metadata(
    source_dir: Path,
    metadata_xlsx: Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Load metadata from Excel and/or source_dir/metadata.json keyed by filename."""
    metadata: dict[str, dict[str, Any]] = {}

    if metadata_xlsx and metadata_xlsx.exists():
        metadata.update(load_metadata_xlsx(metadata_xlsx))

    metadata_path = source_dir / "metadata.json"
    if metadata_path.exists():
        with metadata_path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, dict):
            metadata.update(payload)

    return metadata


def extract_pdf_text(pdf_path: Path) -> tuple[str, list[int]]:
    """Extract full text and page boundaries using PyMuPDF."""
    document = fitz.open(pdf_path)
    pages: list[str] = []
    page_numbers: list[int] = []

    for index, page in enumerate(document, start=1):
        text = page.get_text("text").strip()
        if text:
            pages.append(text)
            page_numbers.append(index)

    document.close()
    return "\n\n".join(pages), page_numbers


def discover_pdfs(pdf_dir: Path) -> list[Path]:
    return sorted(path for path in pdf_dir.rglob("*.pdf") if path.is_file())


def read_text_file(text_path: Path) -> str:
    """Read a cleaned plain-text paper file."""
    return text_path.read_text(encoding="utf-8").strip()


def discover_text_files(text_dir: Path) -> list[Path]:
    return sorted(path for path in text_dir.rglob("*.txt") if path.is_file())
