# backend/core/rag/llm_runner.py

import requests
from typing import Optional

from backend.core.rag.settings import OLLAMA_TIMEOUT_SECONDS, OLLAMA_API_URL


def ask_ollama(prompt: str, model: str, *, extra_params: Optional[dict] = None) -> str:
    """
    Thin wrapper around the Ollama HTTP API.

    - Uses a configurable timeout
    - Returns the "response" field from Ollama's JSON
    - Raises clear RuntimeError on failure
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,  # we want the full answer at once
    }

    if extra_params:
        payload.update(extra_params)

    try:
        resp = requests.post(
            OLLAMA_API_URL,
            json=payload,
            timeout=OLLAMA_TIMEOUT_SECONDS,
        )
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Ollama connection error: {e}") from e

    if resp.status_code != 200:
        raise RuntimeError(f"Ollama error ({resp.status_code}): {resp.text}")

    data = resp.json()

    # For non-streaming requests, Ollama returns a single JSON object with "response"
    if "response" not in data:
        raise RuntimeError(f"Ollama response missing 'response' field: {data}")

    return data["response"]