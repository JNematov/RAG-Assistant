from typing import List, Dict
from time import perf_counter

from .vector_store import get_collection
from backend.services.llm_service import LLMService

_llm = LLMService()


def _load_all_chunks(collection_name: str) -> List[Dict]:
    """
    Load ALL chunks from a given Chroma collection.

    Returns a list of dicts: { 'id', 'text', 'metadata' }
    """
    col = get_collection(collection_name)

    # Chroma returns ids always; we only need to include docs + metadatas
    res = col.get(include=["documents", "metadatas"])

    ids = res.get("ids", [])
    documents = res.get("documents", [])
    metadatas = res.get("metadatas", [])

    docs: List[Dict] = []
    for i in range(len(ids)):
        docs.append({
            "id": ids[i],
            "text": documents[i],
            "metadata": metadatas[i],
        })

    return docs


def _make_batches(
    docs: List[Dict],
    max_chars_per_batch: int = 2500,  # smaller than before for speed
) -> List[str]:
    """
    Group chunk texts into batches so that each batch
    is at most ~max_chars_per_batch characters.

    This helps stay well within the model's context window and
    keeps each LLM call relatively fast.
    """
    batches: List[str] = []
    current_parts: List[str] = []
    current_len = 0

    for doc in docs:
        text = doc["text"].strip()
        if not text:
            continue

        # +2 for the '\n\n' separator
        if current_len + len(text) + 2 > max_chars_per_batch and current_parts:
            # close current batch
            batches.append("\n\n".join(current_parts))
            current_parts = []
            current_len = 0

        current_parts.append(text)
        current_len += len(text) + 2

    if current_parts:
        batches.append("\n\n".join(current_parts))

    return batches


def _summarize_batch(batch_text: str, model: str) -> str:
    """
    Ask the LLM to summarize a single batch of text.

    The prompt is tuned for speed:
    - We tell the model NOT to show its reasoning
    - We ask for concise bullet points
    """
    prompt = f"""You are summarizing my personal technical notes.

You will be given a chunk of my notes. Your job is to:
- Extract the main ideas and concepts
- Group related ideas together
- Use simple, clear language
- Keep it relatively concise but informative

IMPORTANT:
- Do NOT explain your reasoning
- Do NOT think step-by-step
- Directly output the final summary as bullet points

Here is the text to summarize:

---------------- BEGIN TEXT ----------------
{batch_text}
---------------- END TEXT ----------------

Now write a bullet-point summary of the key ideas:
"""

    return _llm.generate(prompt, model=model)

def summarize_collection(
    collection_name: str,
    batch_model: str = "qwen2.5:7b",
    final_model: str = "deepseek-r1:7b",
) -> str:
    """
    High-level function:
    - Load all chunks from a collection
    - Batch them
    - Summarize each batch with a fast model (batch_model)
    - Combine those summaries into one final summary with a (possibly) slower, smarter model (final_model)
    """
    overall_start = perf_counter()

    docs = _load_all_chunks(collection_name)
    if not docs:
        return "No documents found in this collection."

    print(f"[INFO] Summarizer: loaded {len(docs)} chunk(s) from collection '{collection_name}'.")

    # 1) Make batches of raw chunk text (smaller batches for speed)
    batches = _make_batches(docs)
    print(f"[INFO] Summarizer: created {len(batches)} batch(es).")

    # 2) Summarize each batch individually with the fast model
    batch_summaries: List[str] = []
    for i, batch_text in enumerate(batches, start=1):
        print(f"[INFO] Summarizer: summarizing batch {i}/{len(batches)} using model '{batch_model}'...")
        start = perf_counter()
        summary = _summarize_batch(batch_text, model=batch_model)
        end = perf_counter()
        elapsed = end - start
        print(f"[INFO] Summarizer: batch {i} done in {elapsed:.2f} seconds.")
        batch_summaries.append(summary)

    # If there is only one batch and we're using the same model for final summary,
    # we can just return that batch summary directly.
    if len(batch_summaries) == 1 and batch_model == final_model:
        total_elapsed = perf_counter() - overall_start
        print(f"[INFO] Summarizer: completed single-batch summary in {total_elapsed:.2f} seconds.")
        return batch_summaries[0]

    if len(batch_summaries) == 1:
        # Single batch but different final_model: do a light re-summarization
        joined_summaries = batch_summaries[0]
    else:
        # 3) Combine all batch summaries into one final summary input
        joined_summaries = "\n\n".join(
            f"Summary {i}:\n{txt}" for i, txt in enumerate(batch_summaries, start=1)
        )

    final_prompt = f"""You are summarizing my entire knowledge base from multiple partial summaries.

You will be given several summaries that were generated from different parts of my notes.
Your job is to:
- Merge them into a single coherent overview
- Remove duplicates and redundancies
- Organize the ideas logically
- Keep it clear, high-level, and easy to review later

IMPORTANT:
- Do NOT explain your reasoning
- Do NOT show intermediate steps
- Directly output the final clean summary

Here are the partial summaries:

---------------- PARTIAL SUMMARIES ----------------
{joined_summaries}
---------------- END PARTIAL SUMMARIES ----------------

Now write a single, well-structured summary of my notes:
"""

    print(f"[INFO] Summarizer: generating final merged summary using model '{final_model}'...")
    final_start = perf_counter()
    final_summary = _llm.generate(final_prompt, model=final_model)
    final_end = perf_counter()
    print(f"[INFO] Summarizer: final summary done in {final_end - final_start:.2f} seconds.")
    
    total_elapsed = perf_counter() - overall_start
    print(f"[INFO] Summarizer: total summarization time: {total_elapsed:.2f} seconds.")

    return final_summary