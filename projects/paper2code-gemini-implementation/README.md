# Paper2Code Gemini Pipeline

This repository demonstrates how to reproduce the Paper2Code multi-stage workflow (planning → analysis → code generation) using Google's Gemini Generative Language API. You point the service at a research paper (converted to JSON), and it generates a repository skeleton and supporting artifacts that mirror the original Paper2Code structure.

- **LLM**: models/gemini-2.0-flash
- **Pipeline**: All three stages handled in a single FastAPI request
- **Artifacts**: Stored under Paper2Code/outputs_<project> so upstream evaluation scripts can still be run

## Features

- End-to-end HTTP API that takes a S2ORC-style JSON paper, produces planning artifacts, logic analyses, and code files
- Strict validation of planning responses so coding never runs with empty task lists
- Compatible output directories (planning_response.json, coding_artifacts/, etc.) to ease reuse of the official Paper2Code evaluation tooling
- Detailed instructions for converting PDFs to JSON using s2orc-doc2json + Grobid Docker

## Project Layout

```
├── README.md                 # You are here
├── requirements.txt          # Runtime dependencies
├── .env.example              # Template for environment variables
├── data/                     # Optional sample JSON papers
├── src/                      # FastAPI application and Gemini adapter
│   ├── app.py
│   ├── gemini_adapter.py
│   └── main.py
├── Paper2Code/               # Reused assets from the original project
│   ├── README.md             # Notes on what is reused and why
│   ├── codes/
│   │   ├── utils.py
│   │   └── 0_pdf_process.py
│   └── examples/
└── main.py                   # Legacy demo API (no external LLM calls)
```

## Prerequisites

- Python 3.10+
- A valid Gemini API key with access to models/gemini-2.0-flash
- Docker (optional, only required if you want to convert PDFs yourself)

## Quickstart

### 1. Create and activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1

or

uv init

```

### 2. Install dependencies

```powershell
pip install -r requirements.txt

or 

uv sync
```

### 3. Configure environment variables

Copy .env.example to .env and fill in your real Gemini key (and custom base URL if needed)

### 4. (Optional) Convert a PDF to JSON

Use the official s2orc-doc2json repository together with the Grobid Docker image. Below is a concise transcript:

```powershell
git clone https://github.com/allenai/s2orc-doc2json.git
docker pull grobid/grobid:0.7.3
docker run --name grobid -p 8070:8070 -p 8071:8071 grobid/grobid:0.7.3

uv run python -m doc2json.grobid2json.process_pdf 
    -i path/to/paper.pdf -t temp_dir -o data/json/

python Paper2Code/codes/0_pdf_process.py 
    --input_json_path data/json/paper.json 
    --output_json_path data/json/paper_cleaned.json

# When finished
docker stop grobid
docker rm grobid
```

The *-cleaned.json file is what you feed into the Gemini service.

### 5. Run the FastAPI server

```powershell
uv run python -m src.main
# Server listens on http://127.0.0.1:8100 by default
```

### 6. Submit a paper

```powershell
Invoke-WebRequest 
  -Uri "http://127.0.0.1:8100/run" 
  -Method POST 
  -ContentType "application/json" 
  -Body '{"project_name":"wine_quality", "paper_json_path":"data/wine5_cleaned.json"}'
```

You will receive a JSON response pointing at:

- Paper2Code/outputs_wine_quality – planning logs, analyses, raw code
- Paper2Code/outputs_wine_quality_repo – generated Python modules

## How Paper2Code assets are used

We bundle a subset of the upstream Paper2Code repository (see Paper2Code/README.md for a full list). In brief:

- codes/utils.py gives us content_to_json and companion helpers to robustly parse [CONTENT]...[/CONTENT] blocks.
- codes/0_pdf_process.py performs the official Stage-0 JSON cleaning.
- Directory names (outputs_<project>, coding_artifacts/, etc.) match the original convention so evaluation scripts such as codes/eval.py can run unchanged.

## Current limitations & contributions

- The original evaluation step (codes/eval.py) is not yet wired into the FastAPI pipeline. Feel free to open a PR that adds an evaluation endpoint or CLI command.
- Any additional polishing—linting, tests, richer prompts—is welcome. Please document changes clearly so users can understand how they diverge from upstream Paper2Code.

To get started contributing, fork the repo, make your improvements, and submit a pull request. We’re especially interested in:
- Better prompt engineering that produces richer plan structures
- Integrations with alternative LLM providers
- Hooks for the official Paper2Code evaluation scripts

Happy hacking! If you run into questions about the code or data flow, open an issue or start a discussion.
