# Local RAG with LEANN 🔍

Build a local Retrieval-Augmented Generation system using LeANN (Lightweight Approximate Nearest Neighbors) - fast, efficient document search and question answering without external APIs.

## Why This Matters

- 🏠 **Local Processing**: Complete RAG pipeline running locally with no external dependencies
- ⚡ **LeANN Engine**: Lightweight approximate nearest neighbor search for fast retrieval
- 🔒 **Privacy First**: All documents and queries processed locally - data never leaves your machine
- 💰 **Cost Free**: No API costs - unlimited document indexing and querying
- 📚 **Efficient Indexing**: Fast document processing and vector storage

## Key Features

- **Smart Document Retrieval**: Context-aware search through your document collection
- **Local Vector Storage**: Efficient indexing with LeANN's approximate nearest neighbors
- **Real-time Q&A**: Instant answers from your documents
- **Multiple Format Support**: Process various document types
- **Persistent Index**: Save and reload document indexes

## Installation and Setup

### Prerequisites
- Python 3.12 or later
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

### 1. Install Dependencies

**Using uv (recommended):**
```bash
uv sync
```

**Using pip:**
```bash
pip install leann
```

### 2. Run the App

```bash
uv run main.py
# or with pip: python main.py
```

## Project Structure

```
local-rag-leann/
├── main.py                      # Main RAG application
├── demo.index                   # LeANN index file
├── demo.leann.meta.json         # Index metadata
├── demo.leann.passages.idx      # Passage index
├── demo.leann.passages.jsonl    # Passage data
└── README.md
```

## Usage

1. Start the application
2. Add documents to build your knowledge base
3. Ask questions about your documents
4. Get contextual answers with source references
5. Index persists automatically for future sessions

---

## Contribution

Contributions are welcome! Please fork the repository and submit a pull request with your improvements.