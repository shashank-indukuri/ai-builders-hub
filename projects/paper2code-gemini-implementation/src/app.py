"""Gemini-driven orchestrator for Paper2Code demo.

Single-file FastAPI service. Writes artifacts under Paper2Code/outputs_{project_name}.
Requires GEMINI_API_KEY in the environment.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .gemini_adapter import generate_text

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(dotenv_path=BASE_DIR / ".env", override=False)

try:
    from Paper2Code.codes import utils as p2c_utils
except Exception as exc:  # pragma: no cover
    raise RuntimeError("Paper2Code.codes.utils import failed: " + str(exc)) from exc


class PlanValidationError(ValueError):
    """Raised when the planning stage returns an invalid structure."""


app = FastAPI(title="Paper2Code Gemini Demo")


class RunRequest(BaseModel):
    paper_json_path: Optional[str] = None
    project_name: str = "demo"
    model: str = "models/gemini-2.0-flash"


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


def _require_api_key() -> None:
    if os.environ.get("GEMINI_API_KEY") is None:
        raise HTTPException(status_code=400, detail="GEMINI_API_KEY not set")


def _clean_string_list(raw_value: object) -> List[str]:
    cleaned: List[str] = []
    if isinstance(raw_value, list):
        seen = set()
        for item in raw_value:
            if not isinstance(item, str):
                continue
            candidate = item.strip().replace("\\", "/")
            if candidate.startswith("./"):
                candidate = candidate[2:]
            if candidate.endswith("/"):
                candidate = candidate[:-1]
            if candidate and candidate not in seen:
                seen.add(candidate)
                cleaned.append(candidate)
    return cleaned


def _ensure_plan(raw_plan: str, paper_excerpt: str) -> Dict[str, object]:
    try:
        parsed = p2c_utils.content_to_json(raw_plan)
    except Exception as exc:
        raise PlanValidationError("planning output could not be parsed as JSON") from exc

    if not isinstance(parsed, dict):
        raise PlanValidationError("planning output must be a JSON object")

    required_keys = {
        "Implementation approach",
        "File list",
        "Data structures and interfaces",
        "Program call flow",
        "Task list",
    }
    missing = [key for key in required_keys if key not in parsed]
    if missing:
        raise PlanValidationError("planning output missing keys: " + ", ".join(missing))

    file_list = _clean_string_list(parsed.get("File list"))
    if not file_list:
        raise PlanValidationError("File list must contain at least one filename")
    parsed["File list"] = file_list

    task_list = _clean_string_list(parsed.get("Task list"))
    if not task_list:
        task_list = list(file_list)
    parsed["Task list"] = task_list

    impl = parsed.get("Implementation approach")
    if not isinstance(impl, str) or len(impl.strip()) < 20:
        excerpt = paper_excerpt.replace("\r", " ").replace("\n", " ")
        snippet = excerpt[:200] + ("…" if len(excerpt) > 200 else "")
        raise PlanValidationError(
            "Implementation approach must be descriptive; excerpt observed: " + snippet
        )

    ds = parsed.get("Data structures and interfaces")
    if not isinstance(ds, dict):
        raise PlanValidationError("Data structures and interfaces must be an object")
    for fname in file_list:
        description = ds.get(fname)
        if not isinstance(description, str) or not description.strip():
            raise PlanValidationError(f"Data structures and interfaces must describe '{fname}'")

    flow = parsed.get("Program call flow")
    if not isinstance(flow, list) or not all(isinstance(item, str) and item.strip() for item in flow):
        raise PlanValidationError("Program call flow must be a list of non-empty strings")

    return parsed


def _request_plan(model: str, paper_excerpt: str, attempts: int = 3) -> Dict[str, object]:
    prompt_base = (
        "You are an expert software planner. Produce JSON wrapped in [CONTENT] and [/CONTENT] "
        "with exactly the keys Implementation approach, File list, Data structures and interfaces, "
        "Program call flow, Task list.\n"
        "Formatting rules:\n"
        "- Output must be valid JSON (double quotes, no trailing comments).\n"
        "- File list must be an array of relative filenames (no descriptions).\n"
        "- Task list must repeat those filenames in dependency order (no numbering).\n"
        "- Data structures and interfaces must map each filename to a concise implementation note.\n"
        "- Program call flow must be an array of strings describing the execution sequence referencing the filenames.\n"
        "Avoid placeholder text; infer reasonable details from the paper excerpt.\n\n"
        f"Paper excerpt:\n{paper_excerpt[:2000]}\n"
    )

    last_error: Optional[str] = None
    raw_responses: List[str] = []

    for attempt in range(attempts):
        prompt = prompt_base
        if last_error:
            prompt += (
                "\n\nYour previous response was invalid because: "
                + last_error
                + "\nReturn only valid JSON with the required keys."
            )
        raw_plan = generate_text(prompt, model=model, temperature=0.2, max_tokens=1200)
        raw_responses.append(raw_plan)
        try:
            plan = _ensure_plan(raw_plan, paper_excerpt)
            return plan
        except PlanValidationError as exc:
            last_error = str(exc)

    raise PlanValidationError(last_error or "Failed to produce a valid plan")


@app.post("/run")
def run_demo(req: RunRequest) -> Dict[str, str]:
    """Run planning -> analyzing -> coding using Gemini and Paper2Code utils."""
    _require_api_key()

    out_base = Path(f"Paper2Code/outputs_{req.project_name}")
    repo_out = Path(f"Paper2Code/outputs_{req.project_name}_repo")
    (out_base / "analyzing_artifacts").mkdir(parents=True, exist_ok=True)
    (out_base / "coding_artifacts").mkdir(parents=True, exist_ok=True)
    repo_out.mkdir(parents=True, exist_ok=True)

    paper_content = "[NO_PAPER]"
    if req.paper_json_path:
        p = Path(req.paper_json_path)
        if not p.exists():
            raise HTTPException(status_code=404, detail="paper_json_path not found")
        paper_content = p.read_text(encoding="utf-8")

    try:
        plan_data = _request_plan(req.model, paper_content)
    except PlanValidationError as exc:
        raise HTTPException(status_code=500, detail=f"Invalid planning output: {exc}") from exc

    normalized_plan_text = "[CONTENT]\n" + json.dumps(plan_data, indent=2) + "\n[/CONTENT]"
    (out_base / "planning_response.json").write_text(
        json.dumps([{"text": normalized_plan_text}], indent=2), encoding="utf-8"
    )
    (out_base / "task_list.json").write_text(
        json.dumps({"Task list": plan_data["Task list"]}, indent=2), encoding="utf-8"
    )

    logic_analysis: Dict[str, str] = {}
    for fname in plan_data["Task list"]:
        ana_prompt = (
            f"Analyze logic for file {fname}.\nPlan:\n{normalized_plan_text[:800]}"
        )
        ana = generate_text(ana_prompt, model=req.model, temperature=0.2, max_tokens=400)
        logic_analysis[fname] = ana
        (out_base / "analyzing_artifacts" / f"{fname}_analysis.txt").write_text(ana, encoding="utf-8")

    (out_base / "logic_analysis.json").write_text(json.dumps(logic_analysis, indent=2), encoding="utf-8")

    for fname in plan_data["Task list"]:
        code_prompt = (
            f"Implement file {fname}. Return fenced python code block.\nPlan:\n{normalized_plan_text[:800]}\n"
            f"Analysis:\n{logic_analysis.get(fname, '')[:800]}"
        )
        code_resp = generate_text(code_prompt, model=req.model, temperature=0.15, max_tokens=1200)
        (out_base / "coding_artifacts" / f"{fname}_coding.txt").write_text(code_resp, encoding="utf-8")
        match = re.search(r"```python\s*(.*?)```", code_resp, re.DOTALL)
        code_body = match.group(1) if match else code_resp
        target = repo_out / fname
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(code_body, encoding="utf-8")

    return {"outputs_dir": str(out_base), "repo_dir": str(repo_out), "status": "completed"}
