"""Run the src FastAPI app with uvicorn.

Usage:
    python -m src.main
"""
import os
import sys
from pathlib import Path

import uvicorn


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def main():
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", 8100))
    print(f"Starting Paper2Code Gemini service on http://{host}:{port}")
    uvicorn.run("src.app:app", host=host, port=port, log_level="info")


if __name__ == '__main__':
    main()
