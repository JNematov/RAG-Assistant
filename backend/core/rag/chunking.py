from typing import List, Dict

def simple_chunk(text: str, max_chars: int = 800, overlap: int = 200) -> List[str]:
    """
    Basic sliding-window chunking by characters.
    Good enough for v1.
    """
    if len(text) <= max_chars:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + max_chars
        chunk = text[start:end]
        chunks.append(chunk)
        if end >= len(text):
            break
        start = end - overlap
    return chunks


def chunk_documents(docs: List[Dict], max_chars: int = 800, overlap: int = 200) -> List[Dict]:
    """
    Takes list of docs {id, text, metadata} and returns chunk docs:
    {id, text, metadata}
    """
    chunked = []
    for doc in docs:
        chunks = simple_chunk(doc["text"], max_chars=max_chars, overlap=overlap)
        for idx, ch in enumerate(chunks):
            chunked.append({
                "id": f"{doc['id']}#chunk{idx}",
                "text": ch,
                "metadata": {
                    **doc["metadata"],
                    "chunk_index": idx,
                }
            })
    return chunked