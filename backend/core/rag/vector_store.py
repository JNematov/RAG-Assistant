from typing import List, Dict
import chromadb
from .config import CHROMA_DIR, GENERAL_COLLECTION_NAME, CS_COLLECTION_NAME

_client = None

_client = None

def get_client():
    global _client
    if _client is None:
        # Local, on-disk Chroma instance
        _client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return _client


def get_collection(name: str):
    client = get_client()
    return client.get_or_create_collection(name=name)


def clear_collection(name: str):
    client = get_client()
    # If collection exists, delete it entirely, then recreate fresh
    try:
        client.delete_collection(name)
    except Exception:
        # If it doesn't exist yet, ignore
        pass
    # Recreate the empty collection so callers can immediately add to it
    client.get_or_create_collection(name=name)


def add_documents_to_collection(name: str, docs: List[Dict], embeddings):
    col = get_collection(name)
    ids = [d["id"] for d in docs]
    texts = [d["text"] for d in docs]
    metas = [d["metadata"] for d in docs]

    col.add(ids=ids, documents=texts, metadatas=metas, embeddings=list(embeddings))


def query_collection(name: str, query: str, k: int = 5) -> Dict:
    col = get_collection(name)
    return col.query(query_texts=[query], n_results=k)