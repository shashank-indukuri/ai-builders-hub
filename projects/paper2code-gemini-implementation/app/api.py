from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
import json
from typing import List

# Import local utilities from Paper2Code if available
try:
    from Paper2Code.codes import utils
except Exception:
    # Fallback: provide minimal stubs for the utilities we use
    class _StubUtils:
        @staticmethod
        def read_python_files(directory):
            return {}

    utils = _StubUtils()


app = FastAPI(title="Paper2Code Demo API", version="0.1")


class SummarizeResponse(BaseModel):
    files: List[str]


class PlanningCreateRequest(BaseModel):
    project_name: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/summarize-code", response_model=SummarizeResponse)
def summarize_code(path: str = "Paper2Code/codes"):
    """Return a short listing of Python files under the given path.

    This endpoint does not call any external services and only uses local
    file utilities. It purposefully avoids using the `examples/` files.
    """
    p = Path(path)
    if not p.exists():
        raise HTTPException(status_code=404, detail=f"Path not found: {path}")

    py_files = utils.read_python_files(str(p))
    return {"files": list(py_files.keys())}


@app.post("/create-planning-stub")
def create_planning_stub(req: PlanningCreateRequest):
    """Create a small planning stub under `Paper2Code/outputs_{project_name}`.

    This demonstrates writing output artifacts without contacting an LLM.
    """
    base = Path(f"Paper2Code/outputs_{req.project_name}")
    base.mkdir(parents=True, exist_ok=True)

    trajectories = [
        {"role": "system", "content": "[DUMMY] Planning system message"},
        {"role": "user", "content": "[DUMMY] Please plan"},
        {"role": "assistant", "content": "[CONTENT]{\n    \"Implementation approach\": \"Demo approach\",\n    \"File list\": [\"config.yaml\"],\n    \"Data structures and interfaces\": \"None\",\n    \"Program call flow\": \"demo\"\n}[CONTENT]"}
    ]

    response = [{"choices": [{"message": {"content": "[CONTENT]{\n  \"Required packages\": [\"numpy\"],\n  \"Logic Analysis\": [[\"demo.py\", \"Demo file\"]],\n  \"Task list\": [\"demo.py\"]\n}[CONTENT]"}}]}]

    (base / "planning_trajectories.json").write_text(json.dumps(trajectories, indent=2), encoding="utf-8")
    (base / "planning_response.json").write_text(json.dumps(response, indent=2), encoding="utf-8")

    return {"path": str(base), "status": "written"}
