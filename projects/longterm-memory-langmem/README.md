# Long-term Memory with LangMem 🧠

Build AI agents with persistent memory capabilities using LangGraph and LangMem - create conversational AI that remembers context across sessions and interactions.

## Why This Matters

- 🧠 **Persistent Memory**: AI agents that remember user preferences and conversation history
- 🏠 **Local Processing**: Powered by Ollama's local inference with Qwen3:8b model
- 🔒 **Privacy First**: All memory storage happens locally - your data never leaves your machine
- ⚡ **Fast Retrieval**: Vector-based memory search for instant context recall
- 💰 **Cost Free**: No API costs - run unlimited memory operations locally

## Key Features

- **Memory Management**: Store and retrieve user preferences, facts, and context
- **Vector Search**: Semantic search through stored memories using embeddings
- **Persistent Storage**: Memories survive across application restarts
- **LangGraph Integration**: Built on LangGraph's reactive agent framework
- **Local Embeddings**: Uses Ollama's nomic-embed-text for privacy-focused embeddings

## Installation and Setup

### Prerequisites
- Python 3.12 or later
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- [Ollama](https://ollama.ai/) installed and running
- Qwen3:8b and nomic-embed-text models downloaded

### 1. Install Ollama and Models

1. Install Ollama from [ollama.ai](https://ollama.ai/)
2. Pull the required models:

```bash
ollama pull qwen3:8b
ollama pull nomic-embed-text
```

### 2. Install Dependencies

**Using uv (recommended):**
```bash
uv sync
```

**Using pip:**
```bash
pip install langgraph langmem python-dotenv
```

### 3. Run the App

```bash
uv run python main.py
# or with pip: python main.py
```

## Project Structure

```
longterm-memory-langmem/
├── main.py              # Main application with memory agent
├── .env.example         # Environment variables template
└── README.md
```

## Usage

1. Run the application
2. The agent will store a memory about your programming preferences
3. Query the agent about your background - it will recall the stored information
4. Extend the code to add more complex memory interactions


---

## Contribution

Contributions are welcome! Please fork the repository and submit a pull request with your improvements.