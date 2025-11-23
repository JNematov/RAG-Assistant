# backend/services/llm_service.py

import os
from typing import Optional

import requests


class LLMService:
    """
    Abstraction layer over whichever LLM backend you use.

    - Today: Ollama (local DeepSeek / Qwen / etc.)
    - Later: we can add Groq support here without touching orchestrator/router.
    """

    def __init__(self, backend: Optional[str] = None):
        # "ollama" for now, can later be "groq"
        self.backend = backend or os.getenv("LLM_BACKEND", "ollama")

        # Model names are whatever you have in Ollama:
        # e.g. "deepseek-r1:7b", "qwen2.5:7b", etc.
        self.default_model = os.getenv("LLM_DEFAULT_MODEL", "deepseek-r1:7b")

        # Ollama base URL (default local install)
        self.ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")

        # Timeout for HTTP calls (seconds)
        self.timeout = int(os.getenv("OLLAMA_TIMEOUT_SECONDS", "120"))

    def generate(self, prompt: str, model: Optional[str] = None) -> str:
        """
        Public method used by router/orchestrator.
        """
        model_name = model or self.default_model

        if self.backend == "ollama":
            return self._call_ollama(prompt, model_name)
        elif self.backend == "groq":
            # placeholder for future Groq integration
            return self._call_groq(prompt, model_name)
        else:
            raise ValueError(f"Unsupported LLM backend: {self.backend}")

    # ------------------------
    # Ollama backend
    # ------------------------

    def _call_ollama(self, prompt: str, model: str) -> str:
        """
        Call a local Ollama model via HTTP.
        Uses the /api/generate endpoint with streaming disabled.
        """

        url = f"{self.ollama_url}/api/generate"

        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
        }

        resp = requests.post(url, json=payload, timeout=self.timeout)
        resp.raise_for_status()

        data = resp.json()
        # Ollama returns {"model": "...", "created_at": "...", "response": "...", ...}
        return (data.get("response") or "").strip()

    # ------------------------
    # Groq backend (future)
    # ------------------------

    def _call_groq(self, prompt: str, model: str) -> str:
        """
        Placeholder for future Groq integration.
        When you're ready, you'll:
          - pip install groq
          - read GROQ_API_KEY from env
          - call Groq's chat/completions endpoint here.
        """
        raise NotImplementedError("Groq backend not implemented yet.")