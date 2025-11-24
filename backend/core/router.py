# backend/core/router.py

from dataclasses import dataclass
from typing import Any, Dict, List
import json
import logging
import requests

from backend.core.rag.llm_runner import ask_ollama
from backend.core.rag.settings import (
    ROUTER_MODEL,
    USE_GROQ_ROUTER,
    GROQ_API_KEY,
    GROQ_ROUTER_MODEL,
    GROQ_API_URL,
)
from backend.services.llm_service import LLMService


@dataclass
class RouteDecision:
    operation: str            # "qa", "summarize", "email_latest", "free_chat"
    primary_source: str       # "notes", "emails", "documents", "all"
    secondary_sources: List[str]
    arguments: Dict[str, Any]
    reasoning: str
    search_strategy: str
    confidence: float


llm_router = LLMService()  # can pass backend="ollama" explicitly if you want
logger = logging.getLogger(__name__)


def _safe_parse_json(raw: str) -> Dict[str, Any]:
    try:
        start = raw.find("{")
        end = raw.rfind("}")
        snippet = raw[start : end + 1]
        return json.loads(snippet)
    except Exception:
        return {
            "operation": "qa",
            "primary_source": "all",
            "secondary_sources": [],
            "arguments": {},
            "reasoning": "Fallback routing due to parse error.",
            "search_strategy": "Search all sources.",
            "confidence": 0.1,
        }


def route_query(message: str) -> RouteDecision:
    """
    Use either Groq or Ollama to decide:
      - operation (qa, summarize, email_latest, free_chat, etc.)
      - primary_source (notes/emails/documents/all)
      - secondary_sources
      - arguments (e.g. sender name)
      - reasoning, search_strategy
      - confidence
    """

    if USE_GROQ_ROUTER and GROQ_API_KEY:
        raw_json = _route_with_groq(message)
    else:
        raw_json = _route_with_ollama(message)

    # Now construct a RouteDecision from the parsed JSON
    op = raw_json.get("operation", "qa")
    primary = raw_json.get("primary_source", "notes")
    secondary = raw_json.get("secondary_sources", []) or []
    arguments = raw_json.get("arguments", {}) or {}
    reasoning = raw_json.get("reasoning", "")
    strategy = raw_json.get("search_strategy", "")
    confidence = float(raw_json.get("confidence", 0.9))

    return RouteDecision(
        operation=op,
        primary_source=primary,
        secondary_sources=secondary,
        arguments=arguments,
        reasoning=reasoning,
        search_strategy=strategy,
        confidence=confidence,
    )
    routing_prompt = f"""
You are a router for a personal AI knowledge assistant.

Your job is to:
1. Figure out WHAT kind of operation the user wants.
2. Decide WHICH data sources should be searched.
3. Extract any special arguments (like a sender's name).

User query:
\"\"\"{message}\"\"\"


Available operations:
- "qa": answer a specific question using stored data (default)
- "summarize": create a summary from multiple pieces of content
- "email_latest": get and summarize the most recent relevant email
- "free_chat": general conversation, no retrieval required

Available sources:
- "notes": personal notes, learning materials, book takeaways
- "emails": email messages, links, conversations
- "documents": PDFs, books, long-form content
- "all": when the query is broad or unclear

Return STRICT JSON:
{{
  "operation": "qa | summarize | email_latest | free_chat",
  "primary_source": "notes | emails | documents | all",
  "secondary_sources": ["optional other sources"],
  "arguments": {{
    "sender": "<sender name or email if relevant>",
    "scope": "<scope like 'cs' for summarization, if relevant>"
  }},
  "reasoning": "why you chose this operation and these sources",
  "search_strategy": "how to search these sources",
  "confidence": 0.0-1.0
}}
"""

    raw = llm_router.generate(routing_prompt)
    data = _safe_parse_json(raw)

    return RouteDecision(
        operation=data.get("operation", "qa"),
        primary_source=data.get("primary_source", "all"),
        secondary_sources=data.get("secondary_sources", []) or [],
        arguments=data.get("arguments", {}) or {},
        reasoning=data.get("reasoning", ""),
        search_strategy=data.get("search_strategy", ""),
        confidence=float(data.get("confidence", 0.5)),
    )

def _build_routing_prompt(message: str) -> str:
    """
    Shared routing prompt for both Groq and Ollama backends.
    """
    return f"""
You are a routing controller for a personal AI knowledge assistant.

Analyze the user's message and decide:
1. What OPERATION to perform:
   - "qa"          -> answer a specific question using notes/documents
   - "summarize"   -> summarize a set of notes or documents
   - "email_latest"-> fetch/summarize the most recent email from a given sender
   - "free_chat"   -> general conversation without searching

2. Which DATA SOURCES to search:
   - "notes"       -> personal notes, learning materials, book takeaways
   - "emails"      -> email messages, links, conversations
   - "documents"   -> PDFs, long-form content
   - "all"         -> search all sources

SPECIAL RULE FOR NOTES-ONLY QUERIES: If the query mentions "my notes", "i wrote", "in my notes", or refers to specific personal content (e.g., "that i wrote in my linux notes"), set primary_source to "notes" and secondary_sources to an empty list []. Do not add any secondary sources or "all" in these cases—stick strictly to personal notes for exact recall.    

3. Any ARGUMENTS needed:
   - e.g. for "email_latest", extract a "sender" field

Return STRICT JSON with this schema:
{{
  "operation": "qa | summarize | email_latest | free_chat",
  "primary_source": "notes | emails | documents | all",
  "secondary_sources": ["notes", "emails", "documents"],
  "arguments": {{"sender": "Linus Torvalds"}},
  "reasoning": "short explanation of your decision",
  "search_strategy": "how to search the chosen sources",
  "confidence": 0.0 to 1.0
}}

User message:
\"\"\"{message}\"\"\"
"""


def _route_with_ollama(message: str) -> Dict[str, Any]:
    prompt = _build_routing_prompt(message)
    raw = ask_ollama(prompt, model=ROUTER_MODEL)

    # Be resilient to slightly malformed JSON from the model
    data = _safe_parse_json(raw)
    if data.get("reasoning", "").startswith("Fallback routing"):
        logger.warning("Router JSON parse issue; raw response: %s", raw)
    return data


def _route_with_groq(message: str) -> Dict[str, Any]:
    """
    Call Groq's OpenAI-compatible chat.completions API to perform routing.

    Requires:
      - GROQ_API_KEY
      - GROQ_ROUTER_MODEL
    """
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    prompt = _build_routing_prompt(message)

    body = {
        "model": GROQ_ROUTER_MODEL,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        "temperature": 0.1,
        "max_tokens": 256,
    }

    resp = requests.post(GROQ_API_URL, headers=headers, json=body, timeout=20)

    if resp.status_code != 200:
        # If Groq fails, fall back to Ollama routing
        return _route_with_ollama(message)

    data = resp.json()
    try:
        content = data["choices"][0]["message"]["content"]
        parsed = json.loads(content)
    except Exception:
        # If the model doesn't return clean JSON, fall back
        return _route_with_ollama(message)

    return parsed
