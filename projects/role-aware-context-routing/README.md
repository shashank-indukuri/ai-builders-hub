# Role-Aware Context Routing (RCR-Router) 🧠

Efficiently route only the most relevant context to each agent in a multi-agent LLM system. Reduce token usage, improve latency, and keep agents focused on their roles.

## Why This Matters

- 🚨 Multi-agent systems often waste tokens by broadcasting full history to every agent
- 🔎 Rigid routing can’t adapt as tasks evolve, causing noise and higher cost
- ✅ RCR-Router dynamically selects role- and stage-relevant memory under a token budget

From our experiments and paper-aligned heuristics:
- ✅ 33% fewer tokens used
- ⚡ 23% faster responses
- 📈 Equal or better answer quality across benchmarks

Paper inspiration: "RCR-Router: Efficient Role-Aware Context Routing for Multi-Agent LLM Systems with Structured Memory"

## Key Features

- Role-Aware Context Routing with token budgets per role and stage
- Structured Memory with conflict resolution, versioning, and confidence tracking
- BERT-backed or sentence-transformer semantic similarity for relevance scoring
- Knapsack Optimization (exact for small sets; greedy for scale) to fit budgets
- Flexible LLM Providers with automatic fallback: Gemini, Groq, Ollama
- AutoGen-compatible multi-agent orchestration (planner → coder → reviewer)
- Built-in Analytics and Visualization (token savings, per-agent metrics, architecture diagrams)

## Installation and Setup

### Prerequisites
- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- Optional: [Ollama](https://ollama.com) running locally if you want local models
- API keys if you use hosted providers:
  - Gemini (Google AI Studio)
  - Groq ([console.groq.com](https://console.groq.com/))

### 1) Configure Environment
Create a `.env` file in the project root. You can copy from `env.example.env`:

```bash
GEMINI_API_KEY="your_gemini_key_here"
GROQ_API_KEY="your_groq_key_here"
# Optional if your Ollama is not at the default URL
OLLAMA_URL=http://localhost:11434
```

Notes:
- The code loads environment variables via `python-dotenv`. If you see `ModuleNotFoundError: dotenv`, install `python-dotenv` (see below).
- OpenAI/Anthropic are stubbed for compatibility and not required.

### 2) Install Dependencies

Using uv (recommended):
```bash
uv sync
# If you encounter a dotenv import error:
uv add python-dotenv
```

Using pip:
```bash
pip install -e .
pip install python-dotenv
```

Optional (for local inference):
```bash
# Install Ollama and pull a model used by default code paths
# Example model used: qwen2.5vl:latest
ollama pull qwen2.5vl:latest
```

### 3) Quick Provider Health Check
Verify your keys and connectivity:
```bash
uv run python flexible_llm_provider.py
# or with pip
python flexible_llm_provider.py
```
This tests Gemini, Groq, and Ollama connectivity and prints basic latency and token info.

## Run the Demos

### A) Core Multi-Agent Demo (Planner → Coder → Reviewer)
Runs the flexible multi-agent session with role-aware context routing and provider fallback.
```bash
uv run python flexible_autogen_rcr_integration.py
# or
python flexible_autogen_rcr_integration.py
```
What it does:
- Constructs a `SharedMemory` with semantic scoring and token budgets
- Creates three agents with roles and preferred providers
- Injects role-scoped context using the RCR router and runs several rounds
- Prints session stats (rounds, quality, provider availability)

### B) Analytics-Enhanced Demo
Generates side-by-side comparisons and visualizations of Broadcast-All vs RCR routing.
```bash
uv run python rcr_analytics.py
# or
python rcr_analytics.py
```
Outputs:
- `report/` with visuals:
  - `rcr_architecture.png` (architecture diagram)
  - `token_savings.png` (metrics comparison)
  - `agent_performance.png` (per-agent dashboard)
  - `summary.md` (copyable highlights)
- `rcr_traces/` with CSVs of turns and breakdowns
- `langsmith_traces.json` (trace-style export)

## Programmatic Usage

Minimal example to run your own task end-to-end:
```python
from flexible_autogen_rcr_integration import (
    setup_flexible_rcr_system, run_flexible_multi_agent_session
)
from enhanced_rcr_router import SharedMemory, ImportanceWeights

# Setup providers (respects .env: GEMINI_API_KEY, GROQ_API_KEY, OLLAMA_URL)
manager, preferences = setup_flexible_rcr_system(
    provider_preferences={
        "planner": "gemini",   # provider names: "gemini" | "groq" | "ollama"
        "coder": "groq",
        "reviewer": "ollama",
    }
)

# Configure memory and token budgets per role
weights = ImportanceWeights(
    role_relevance=0.8,
    stage_priority=0.6,
    recency=0.4,
    semantic_similarity=1.2,
    decision_boost=1.0,
    plan_boost=0.6,
)
memory = SharedMemory(weights)
role_budgets = {"planner": 2000, "coder": 1500, "reviewer": 1200}

# Run
result = run_flexible_multi_agent_session(
    task="Build a robust CSV processing pipeline with validation and stats.",
    llm_manager=manager,
    max_rounds=8,
    memory=memory,
    role_budgets=role_budgets,
    provider_preferences=preferences,
)
print(result["average_quality"], result["total_rounds"])  # access key stats
```

Important: when customizing `provider_preferences`, use provider names added to the manager (`"gemini"`, `"groq"`, `"ollama"`). Passing model IDs (e.g., `"gemini-2.0-flash"`) as preferences won’t select the exact client and will fall back automatically.

## Project Structure

```
role-aware-context-routing/
├── flexible_autogen_rcr_integration.py  # Multi-agent orchestration + RCR injection + demo
├── enhanced_rcr_router.py               # SharedMemory, ImportanceWeights, knapsack routing, packing
├── flexible_llm_provider.py             # Gemini/Groq/Ollama clients + manager + health checks
├── rcr_analytics.py                     # Analytics-enabled manager and report generation
├── analytics_harness.py                 # Metrics collection, comparisons, trace export
├── visualization_suite.py               # Architecture & performance visualizations
├── env.example.env                      # Example environment variables
├── pyproject.toml                       # Dependency spec (Python 3.12+)
└── README.md                            # This file
```

## How It Works

- __Structured Memory (`SharedMemory`)__
  - Stores `MemoryItem`s with type (`decision`, `plan`, `entity`, `fact`, `tool_trace`), confidence, tokens, and tags
  - Embeddings via BERT (fallback: `sentence-transformers`) for semantic similarity
  - Conflict resolution and versioning for updates

- __Importance Scoring (`ImportanceWeights`)__
  - Combines role relevance, stage priority, recency, semantic similarity, type boosts, and confidence
  - Produces `importance_score` used for selection

- __Routing under Budget (`query_routed`)__
  - Scores all memory items, sorts them, and selects under a token budget
  - Uses exact knapsack for small sets and greedy for scale

- __Context Injection (`pack_context` + `JSON_SCHEMA_INSTRUCTION`)__
  - Injects a role-scoped, budgeted context block as a system message before the agent’s turn
  - Enforces JSON-only structured outputs for reliable extraction

- __Agent Coordination__
  - Stages: `plan` → `execute` → `review`
  - Roles: `planner`, `coder`, `reviewer` with stage advancement logic

- __Providers & Fallback (`LLMProviderManager`)__
  - Add providers: `gemini`, `groq`, `ollama`
  - Auto-fallback across providers in preferred order

## Configuration Tips

- __Provider Preferences__: Use provider keys, not raw model names. Example:
  - `{"planner": "gemini", "coder": "groq", "reviewer": "ollama"}`
- __Ollama__: Ensure `OLLAMA_URL` points to your daemon and the model (e.g. `qwen2.5vl:latest`) is pulled
- __Large dependencies__: `torch`, `transformers` may be large; CPU-only installs are fine for the demo

## Troubleshooting

- __dotenv import error__
  - Install: `uv add python-dotenv` or `pip install python-dotenv`

- __BERT model download slow/failing__
  - The router falls back to `sentence-transformers/all-MiniLM-L6-v2` automatically

- __Groq 401 / Gemini auth error__
  - Double-check `GROQ_API_KEY` / `GEMINI_API_KEY` in `.env`

- __Ollama connection error__
  - Ensure Ollama is running and `OLLAMA_URL` is correct; pull the requested model

- __Provider not selected as expected__
  - Make sure `provider_preferences` use provider names (`"gemini"`, `"groq"`, `"ollama"`) not model IDs

## Contribution

Contributions are welcome! Please open an issue or submit a PR with a clear description, tests (if applicable), and screenshots for analytics/visual changes.

## Citation

If you use this work, please reference the underlying idea:

"RCR-Router: Efficient Role-Aware Context Routing for Multi-Agent LLM Systems with Structured Memory"
