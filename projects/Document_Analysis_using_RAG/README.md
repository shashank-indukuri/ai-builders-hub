# Document Analysis using RAG 🧾

Perform document understanding with a Retrieval-Augmented Generation (RAG) flow: index documents, run retrieval, and generate answers grounded in your documents.

## Why this project

- 🔎 Improve answers with context from your documents (less hallucination)
- ⚡ Fast local experimentation using lightweight tools and `uv`
- 🧩 Composable: indexers and retrievers can be swapped or extended

## Key features

- Document ingestion and simple vector index
- Retrieval + generative answer pipeline (RAG)
- Example `main.py` to demo search and Q&A over loaded documents

## Requirements & prerequisites

- Python 3.10 or later
- `uv` package manager (recommended) or pip
- Google API key

### 1. Get Google API Key

1. Create a `.env` file in the project root:

```bash
GOOGLE_API_KEY="your_google_api_key_here"
```

## Quick start (Windows PowerShell)

```powershell
uv run streamlit run main.py
```

## Project structure

```
Document_Analysis_using_RAG/
|__ .env               # .env file
├── main.py            # example runner for ingestion + query
├── pyproject.toml     # dependency manifest
├── uv.lock            # lockfile for uv
└── README.md
```

## Usage

1. Place documents to index in the project folder or modify `main.py` to point at your data path.
2. Run the ingestion step (handled by `main.py` in the example).
3. Use the interactive prompt or Streamlit UI (if provided) to enter questions; answers will be retrieved and generated from the indexed documents.

## Troubleshooting

- `uv` not found after installation: try `python -m uv` (e.g. `python -m uv run python main.py`) or ensure Python's Scripts folder is on PATH.
- Dependency install fails: try `uv update` or inspect `pyproject.toml` / `uv.lock` for conflicts.

## Contributing

Improvements welcome — open issues or submit a pull request with suggested changes.
