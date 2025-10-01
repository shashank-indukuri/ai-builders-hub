"""Entrypoint for running the demo FastAPI server with uvicorn.

This script starts a small API that demonstrates offline utilities from the
`tests/Paper2Code` package without calling any external LLM services.

Run with:
    python main.py

The server will be available at http://127.0.0.1:8000 by default.
"""
import os
import uvicorn


def main():
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "127.0.0.1")
    print(f"Starting Paper2Code demo API on http://{host}:{port}")
    uvicorn.run("app.api:app", host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
