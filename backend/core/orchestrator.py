# backend/core/orchestrator.py

from typing import Dict, Any, List
import asyncio
from typing import Optional
from backend.core.router import route_query, RouteDecision
from backend.core.rag.query_engine import query_sources
from backend.core.rag.llm import build_prompt_from_hits
from backend.core.rag.summarizer import summarize_collection
from backend.core.rag.config import CS_COLLECTION_NAME, GENERAL_COLLECTION_NAME
from backend.core.rag.settings import DEFAULT_QUERY_MODEL, DEFAULT_BATCH_MODEL, DEFAULT_FINAL_MODEL
from backend.core.rag.llm_runner import ask_ollama


def handle_prompt(message: str) -> Dict[str, Any]:
    """
    Main entry point for the backend.

    - Called by the /prompt endpoint
    - Uses an LLM router (fast model) to decide operation + sources
    - Executes the appropriate flow (email_latest, summarize, or RAG QA/free chat)
    """

    # 1) Route the query (which operation? which sources?)
    route: RouteDecision = route_query(message)
    op = route.operation or "qa"

    # If router isn't confident, treat as free chat / fallback
    if route.confidence is not None and route.confidence < 0.1:
        return _free_chat_response(message, route)

    # 2) Special-case operations
    if op == "email_latest":
        # For now, stubbed out to keep things compiling.
        # Later you can wire this to your IMAP-based email pipeline.
        return _handle_email_latest(route)

    if op == "summarize":
        return _handle_summarize(route)

    if op == "free_chat":
        return _free_chat_response(message, route)

    # 3) Default: intelligent RAG QA
    return _handle_rag_qa(message, route)

async def handle_prompt_async(message: str) -> Dict[str, Any]:
    """
    Async version of handle_prompt that parallelizes:
      - router LLM (route_query)
      - initial vector search (prefetching hits from CS collection)

    The prefetch is only used for the common 'notes/CS' QA path.
    Everything else (summarize, email_latest, free_chat) behaves identically
    to the sync version, just wrapped in async.
    """
    loop = asyncio.get_running_loop()

    # Run routing + initial retrieval in parallel
    routing_task = loop.run_in_executor(None, route_query, message)
    prefetch_task = loop.run_in_executor(None, _prefetch_hits_for_message, message)

    route, prefetched_hits = await asyncio.gather(routing_task, prefetch_task)
    op = route.operation or "qa"

    # Low-confidence → fallback free chat
    if route.confidence is not None and route.confidence < 0.1:
        return _free_chat_response(message, route)

    # Special cases (these don't really benefit from prefetch)
    if op == "email_latest":
        return _handle_email_latest(route)

    if op == "summarize":
        return _handle_summarize(route)

    if op == "free_chat":
        return _free_chat_response(message, route)

    # Default: RAG QA, but try to reuse prefetched hits
    return _handle_rag_qa_with_prefetch(message, route, prefetched_hits)

# ---------------------------------------------------------------------------
#  Helper: RAG QA path
# ---------------------------------------------------------------------------

def _handle_rag_qa(message: str, route: RouteDecision) -> Dict[str, Any]:
    """
    Handle QA-style operations using vector search + LLM.

    - Uses the router's primary + secondary sources to decide which collections to query.
    - Merges hits from multiple collections.
    - Builds a prompt from hits and calls the main QA model.
    """
    rag_sources = _map_sources_to_rag_keys(route)

    all_hits: List[Any] = []

    # Multi-source search: e.g. notes + general documents
    for rag_source in rag_sources:
        hits = query_sources(rag_source, message, k=3)
        all_hits.extend(hits)

    # (Optional) You could deduplicate hits here by doc id or path
    hits_for_prompt = all_hits[:5]

    # Build prompt based on operation (qa vs extract vs whatever, for now just qa)
    prompt = _build_operation_prompt(message, hits_for_prompt, route.operation)

    answer = ask_ollama(prompt, model=DEFAULT_QUERY_MODEL)

    return {
        "answer": answer,
        "sources": hits_for_prompt,
        "routing": route.__dict__,
    }


# ---------------------------------------------------------------------------
#  Helper: Summarize path
# ---------------------------------------------------------------------------

def _handle_summarize(route: RouteDecision) -> Dict[str, Any]:
    """
    Handle summarize operations.

    Right now we only support summarizing the CS collection, but you can
    expand this later to use route.primary_source / secondary_sources.
    """
    # Simple default: summarize the CS knowledge base
    collection_name = CS_COLLECTION_NAME

    summary = summarize_collection(
        collection_name=collection_name,
        batch_model=DEFAULT_BATCH_MODEL,
        final_model=DEFAULT_FINAL_MODEL,
    )

    return {
        "answer": summary,
        "sources": [],
        "routing": route.__dict__,
    }


# ---------------------------------------------------------------------------
#  Helper: Email path (stub for now)
# ---------------------------------------------------------------------------

def _handle_email_latest(route: RouteDecision) -> Dict[str, Any]:
    """
    Handle 'email_latest' operation.

    For now, this is a stub so your backend doesn't crash if the router
    predicts 'email_latest'. Later, you can wire this up to your IMAP
    adapter / Gmail integration and return a proper summary.
    """
    sender = route.arguments.get("sender", "(unknown sender)") if route.arguments else "(unknown sender)"

    answer = (
        f"(email_latest stub) I detected that you want the latest email from '{sender}', "
        f"but the email integration is not wired into the orchestrator yet."
    )

    return {
        "answer": answer,
        "sources": [],
        "routing": route.__dict__,
    }


# ---------------------------------------------------------------------------
#  Helper: Free chat fallback
# ---------------------------------------------------------------------------

def _free_chat_response(message: str, route: RouteDecision) -> Dict[str, Any]:
    """
    Fallback: treat message as normal chat with the main model.
    """
    prompt = (
        "You are my personal assistant. "
        "Answer the following question as helpfully and clearly as possible:\n\n"
        f"{message}"
    )

    answer = ask_ollama(prompt, model=DEFAULT_QUERY_MODEL)

    return {
        "answer": answer,
        "sources": [],
        "routing": route.__dict__,
    }


# ---------------------------------------------------------------------------
#  Helper: mapping LLM-level sources → RAG collection keys
# ---------------------------------------------------------------------------

def _map_sources_to_rag_keys(route: RouteDecision) -> List[str]:
    """
    Map high-level router sources (notes, emails, documents, all)
    to concrete RAG "source keys" used by query_sources.
    """

    # RAG "source" keys used by query_sources, e.g.:
    #   - "cs"       -> CS_COLLECTION_NAME
    #   - "general"  -> GENERAL_COLLECTION_NAME
    #   - "emails"   -> future email collection
    #
    # We'll keep this simple for now and only support:
    #   - notes  -> "cs"
    #   - documents / general -> "general"
    #   - all -> ["cs", "general"]
    primary = (route.primary_source or "notes").lower()
    secondary = [s.lower() for s in (route.secondary_sources or [])]

    rag_keys: List[str] = []

    def add_if_missing(k: str) -> None:
        if k not in rag_keys:
            rag_keys.append(k)

    if primary in ("notes", "cs"):
        add_if_missing("cs")
    if primary in ("documents", "general"):
        add_if_missing("general")
    if primary == "all":
        add_if_missing("cs")
        add_if_missing("general")

    for src in secondary:
        if src in ("notes", "cs"):
            add_if_missing("cs")
        if src in ("documents", "general"):
            add_if_missing("general")
        if src == "all":
            add_if_missing("cs")
            add_if_missing("general")

    # Fallback to CS if nothing resolved
    if not rag_keys:
        rag_keys = ["cs"]

    return rag_keys


# ---------------------------------------------------------------------------
#  Helper: build operation-aware prompt
# ---------------------------------------------------------------------------

def _build_operation_prompt(message: str, hits: List[Any], operation: str | None) -> str:
    """
    For now we only distinguish between 'qa' and everything else.
    Later you can add special prompts for e.g. 'extract', 'plan', etc.
    """
    op = (operation or "qa").lower()

    # Right now, both paths use the same underlying builder, but you could
# tweak the instructions depending on op.~
    if op == "summarize":
        # You could add explicit "summarize these snippets" instructions here.
        return build_prompt_from_hits(
            f"Summarize the key points relevant to: {message}",
            hits,
        )

    # Default: QA style
    return build_prompt_from_hits(message, hits)

def _prefetch_hits_for_message(message: str) -> List[Any]:
    """
    Prefetch hits from the CS collection while the router is running.

    This is a heuristic: most of your questions right now are about your CS notes.
    Later, you can expand this to 'all' or multiple collections if needed.
    """
    try:
        # k=5 is enough for a first QA answer; adjust if needed
        hits = query_sources("cs", message, k=5)
    except Exception:
        hits = []
    return hits


def _handle_rag_qa_with_prefetch(
    message: str,
    route: RouteDecision,
    prefetched_hits: Optional[List[Any]],
) -> Dict[str, Any]:
    """
    RAG QA that *tries* to reuse prefetched hits for the common
    case of notes/CS-only questions. If the router decides other sources
    are needed, we fall back to the normal multi-source retrieval.
    """
    rag_keys = _map_sources_to_rag_keys(route)

    # If the router says: only CS/notes, and we have prefetched hits -> use them
    if rag_keys == ["cs"] and prefetched_hits:
        hits_for_prompt = prefetched_hits
    else:
        # Fallback: do the usual multi-source retrieval
        all_hits: List[Any] = []
        for rag_source in rag_keys:
            hits = query_sources(rag_source, message, k=3)
            all_hits.extend(hits)
        hits_for_prompt = all_hits[:5]

    prompt = _build_operation_prompt(message, hits_for_prompt, route.operation)
    answer = ask_ollama(prompt, model=DEFAULT_QUERY_MODEL)

    return {
        "answer": answer,
        "sources": hits_for_prompt,
        "routing": route.__dict__,
    }