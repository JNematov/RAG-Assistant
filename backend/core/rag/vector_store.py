from typing import List, Dict
import chromadb

from .config import CHROMA_DIR, GENERAL_COLLECTION_NAME, CS_COLLECTION_NAME

_client = None


def get_client():
    """
    Return a chromadb client instance.

    Newer versions of chromadb expose `PersistentClient(path=...)`.
    Older/stable installs typically use `chromadb.Client(Settings(...))`.

    This helper tries the common constructors so the code works across
    different chromadb versions.
    """
    global _client
    if _client is not None:
        return _client

    # Prefer PersistentClient when available (newer API)
    if hasattr(chromadb, "PersistentClient"):
        try:
            _client = chromadb.PersistentClient(path=str(CHROMA_DIR))
            return _client
        except Exception:
            # Fall through to other approaches
            pass

    # Fall back to chromadb.Client with Settings if possible
    try:
        from chromadb.config import Settings

        _client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=str(CHROMA_DIR)))
        return _client
    except Exception:
        pass

    # Last resort: try the plain Client() constructor
    try:
        _client = chromadb.Client()
        return _client
    except Exception as e:
        raise RuntimeError(f"Could not create chromadb client: {e}") from e


def get_collection(name: str):
    client = get_client()
    # try a couple of common method names depending on chromadb version
    if hasattr(client, "get_or_create_collection"):
        return client.get_or_create_collection(name=name)

    # older API fallback
    return client.get_collection(name)


def clear_collection(name: str):
    """Remove all data from a collection (works across chromadb versions).

    We try a few different APIs:
      - client.delete_collection(name)
      - collection.delete(where={})
    and finally ensure the collection exists (empty) afterwards.
    """
    client = get_client()

    # Try to delete the collection entirely (some APIs support this)
    try:
        # try positional
        client.delete_collection(name)
    except TypeError:
        try:
            # try keyword
            client.delete_collection(name=name)
        except Exception:
            pass
    except Exception:
        # ignore other delete errors and try to wipe via collection
        pass

    # If delete_collection isn't available or failed, try deleting documents
    try:
        col = get_collection(name)
        if hasattr(col, "delete"):
            try:
                # prefer the where={} form to delete all docs
                col.delete(where={})
            except TypeError:
                # some versions accept no args
                col.delete()
    except Exception:
        # If everything fails, ignore - callers will recreate collection below
        pass

    # Ensure an empty collection exists for immediate use
    try:
        if hasattr(client, "get_or_create_collection"):
            client.get_or_create_collection(name=name)
        else:
            # some clients have get_collection that creates if missing
            client.get_collection(name)
    except Exception:
        # give up silently; higher-level code will see errors when adding
        pass


def add_documents_to_collection(name: str, docs: List[Dict], embeddings):
    col = get_collection(name)
    ids = [d["id"] for d in docs]
    texts = [d["text"] for d in docs]
    metas = [d["metadata"] for d in docs]

    # Different collection APIs expose add(...) with these named args
    # Keep it simple and call with the most common signature.
    col.add(ids=ids, documents=texts, metadatas=metas, embeddings=list(embeddings))


def query_collection(name: str, query: str, k: int = 5) -> Dict:
    col = get_collection(name)
    # Common query signature
    try:
        return col.query(query_texts=[query], n_results=k)
    except TypeError:
        # older/newer clients may use different arg names
        return col.query(query_texts=[query], n_results=k)