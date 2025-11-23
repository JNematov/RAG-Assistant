# backend/core/rag/llm.py

from typing import List, Dict, Any


def build_prompt_from_hits(
    question: str,
    hits: List[Dict[str, Any]],
    max_context_chars: int = 6000,
) -> str:
    """
    Build a single prompt string using the retrieved chunks.

    Each `hit` is expected to look something like:
        {
            "document": "chunk text ...",
            "metadata": {
                "source": "cs",
                "file": "glossary.txt",
                "chunk_index": 5,
                ...
            },
            "score": 1.23,
        }

    We:
    - include basic metadata so you know where info came from
    - truncate total context to at most `max_context_chars` to keep prompts sane
    """

    if not hits:
        return (
            f"You are my personal knowledge assistant.\n\n"
            f"User question:\n{question}\n\n"
            "No relevant context snippets were found in my personal knowledge base.\n"
            "If you cannot confidently answer, say you are unsure.\n"
        )

    context_parts = []
    total_chars = 0

    for idx, hit in enumerate(hits):
        # Some callers use "text" instead of "document" for the chunk contents
        doc_text = hit.get("document") or hit.get("text") or ""
        meta = hit.get("metadata") or {}

        source = meta.get("source", "unknown")
        file = (
            meta.get("file")
            or meta.get("filename")
            or meta.get("path")
            or meta.get("filepath")
            or "unknown"
        )
        chunk_index = meta.get("chunk_index", idx)

        header = f"[{idx}] (source={source}, file={file}, chunk={chunk_index})"
        snippet = f"{header}\n{doc_text.strip()}\n"

        # Stop if adding this would exceed our max_context_chars
        if total_chars + len(snippet) > max_context_chars:
            break

        context_parts.append(snippet)
        total_chars += len(snippet)

    context_str = "\n\n".join(context_parts)

    prompt = f"""You are my personal knowledge assistant.

You MUST use ONLY the context snippets below to answer the question.
If the answer is unclear or not present in the context, say that you are unsure
and do NOT hallucinate.

Context:
--------
{context_str}

User question:
--------------
{question}

Answer in a clear, step-by-step way, using simple language and concrete examples.
If relevant, reference the snippets you used by their [index] (e.g., [0], [2]).
"""

    return prompt
