# mcpSLM

MCP paper retrieval system for LLP WaterScope-AI project.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install python-dotenv
pip install -r requirements-local.txt
cp .env.example .env
```

First run downloads embedding models from HuggingFace (~100MB).

## Usage

Search (pre-built index included):

```bash
python local_rag_cli.py search "groundwater salinization under climate change"
```

Re-index from your own `.txt` folder:

```bash
python local_rag_cli.py index /path/to/txt/folder --metadata sample_metadata.xlsx
python local_rag_cli.py search "your query" --top-k 10
```
