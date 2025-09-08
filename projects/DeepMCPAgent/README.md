# Repo Assistant with DeepMCPAgent 🔎
Explore and query your codebase using MCP-discovered tools — list directories, grep for patterns, and read files with a model-agnostic agent that works with Gemini or OpenAI.

## Why This Matters
- 🧰 Zero Tool Wiring: Tools are discovered dynamically from MCP servers — no hardcoding
- 🏠 Local-First Tools: The file exploration server runs locally for privacy-safe repo access
- 🧠 Bring Your Own Model: Use Gemini or OpenAI (or swap in your favorite LangChain chat model)
- 🖥️ Clear Traces: Rich console output with discovered tools, per-call traces, and final answers
- 💸 Cost Control: Local file processing is free; choose your preferred model provider

## Key Features
- Smart Repo Exploration: list_dir, grep, and read_file tools for fast code exploration
- Clean Interface: professional console panels/tables via Rich
- Instant Start: examples included; just install and run
- Real-time Agent Trace: see every tool call and outcome

## Installation and Setup
### Prerequisites
- Python 3.11 or later
- pip (or uv if you prefer)
- Optional: API key for Gemini (GEMINI_API_KEY or GOOGLE_API_KEY) or OpenAI (OPENAI_API_KEY)

### Install Dependencies
Using uv (recommended):

```bash
uv sync
```

Using pip:

```bash
pip install -r requirements.txt
```

### Configure Environment
Copy the example env and set your keys as needed:

```bash
# PowerShell (Windows)
Copy-Item .env.example .env
# macOS/Linux
cp .env.example .env
```

Set your API keys in the .env file:

```bash
GEMINI_API_KEY=your_gemini_api_key
GOOGLE_API_KEY=your_google_api_key
OPENAI_API_KEY=your_openai_api_key
```

### Run the Demos
Files demo (local repo assistant):

```bash
# Terminal 1: start the Files MCP server
python examples/servers/files_server.py

# Terminal 2: run the agent that uses the file tools
python examples/use_files_agent.py
```
The file server runs at http://127.0.0.1:8001/mcp

## Project Structure:

DeepMcpAgents/
├── examples/
│   ├── servers/
│   │   ├
│   │   └── files_server.py      # Repo tools: list_dir, grep, read_file (JSON output)
│   ├
│   └── use_files_agent.py       # Repo assistant agent (OpenAI/Gemini)
├
├── .env                         # Your actual keys (do NOT commit)
├── pyproject.toml               # Project metadata and dependencies
├── requirements.txt             # Dependency pins for quick install
└── README.md


## Usage
Files agent (repo assistant):

- Start the Files server: python examples/servers/files_server.py
- Run the agent: python examples/use_files_agent.py
Try prompts like:
- List up to 20 items under the repository root.
- Search for occurrences of 'ChatGoogleGenerativeAI' in Python files under the repo.
- Show the first 60 lines of examples/servers/files_server.py.

## Notes:

- The file tools return JSON; the agent pretty-prints tool results for clarity.
- You can set FILES_SERVER_ROOT before launching the Files server to sandbox to a specific directory.

## Contribution
Contributions are welcome! Please fork the repository and submit a pull request with your improvements.