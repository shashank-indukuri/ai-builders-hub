"""Simple Gemini adapter.

Minimal wrapper around the Gemini Generative Language REST API using
`models/gemini-2.0-flash`. Requires the API key in `GEMINI_API_KEY` and optionally
`GEMINI_BASE_URL` (default `https://generativelanguage.googleapis.com/v1beta`).
"""

from typing import Optional
import json
import os

import requests


API_KEY = os.environ.get("GEMINI_API_KEY")
BASE_URL = os.environ.get("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta")
DEFAULT_MODEL = "models/gemini-2.0-flash"


class GeminiAPIError(RuntimeError):
    """Raised when the Gemini API returns a non-success status code."""


class GeminiConfigError(RuntimeError):
    """Raised when required configuration (API key) is missing."""


def _build_request_body(prompt: str, temperature: float, max_tokens: int) -> dict:
    return {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
        },
    }


def _extract_text(data: dict) -> str:
    candidates = data.get("candidates") or []
    if not candidates:
        return json.dumps(data)

    candidate = candidates[0]
    content = candidate.get("content") or {}
    parts = content.get("parts") or []
    if parts and isinstance(parts[0], dict):
        return parts[0].get("text", "")

    for key in ("output", "outputText", "text"):
        value = candidate.get(key) or data.get(key)
        if isinstance(value, str):
            return value

    return json.dumps(data)


def generate_text(
    prompt: str,
    *,
    model: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 1024,
) -> str:
    """Generate text using the Gemini 2.0 Flash REST API."""
    if API_KEY is None:
        raise GeminiConfigError("GEMINI_API_KEY is not set in the environment")

    chosen_model = model or DEFAULT_MODEL
    if not chosen_model.startswith("models/"):
        chosen_model = f"models/{chosen_model}"

    # Change 1: Pass API_KEY as a query parameter
    url = f"{BASE_URL}/{chosen_model}:generateContent?key={API_KEY}"
    headers = {
        # Change 2: Remove the Authorization header
        "Content-Type": "application/json",
    }

    body = _build_request_body(prompt, temperature, max_tokens)

    response = requests.post(url, headers=headers, data=json.dumps(body), timeout=60)
    if response.status_code != 200:
        raise GeminiAPIError(f"Gemini API call failed: {response.status_code} {response.text}")

    try:
        payload = response.json()
    except Exception as exc:
        raise GeminiAPIError("Invalid JSON response from Gemini API") from exc

    return _extract_text(payload)
