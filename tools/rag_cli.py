"""
Simple CLI for ingesting and querying your notes without running the full backend.

Examples:
  # Ingest
  python tools/rag_cli.py ingest --source cs --clear
  python tools/rag_cli.py ingest --source general

  # Query
  python tools/rag_cli.py query --source all --question "What is Docker networking?"
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List

from sentence_transformers import SentenceTransformer

# Make backend imports work when executed as a script
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.core.rag.config import (  # noqa: E402
    CS_DIR,
    GENERAL_DIR,
    CS_COLLECTION_NAME,
    GENERAL_COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
)
from backend.core.rag.vector_store import (  # noqa: E402
    add_documents_to_collection,
    clear_collection,
    query_collection,
)


# ------------------------
# Loading and chunking
# ------------------------

def _load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _load_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("pypdf is required to ingest PDFs. Install with: pip install pypdf") from exc

    reader = PdfReader(str(path))
    return "\n".join((page.extract_text() or "") for page in reader.pages)


def load_documents(root: Path, source_label: str) -> List[Dict]:
    """
    Load files from a directory into a list of {id, text, metadata}.
    Supports .txt, .md, .pdf.
    """
    docs: List[Dict] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue

        suffix = path.suffix.lower()
        if suffix in (".txt", ".md"):
            text = _load_text(path)
        elif suffix == ".pdf":
            text = _load_pdf(path)
        else:
            continue

        text = text.strip()
        if not text:
            continue

        rel = path.relative_to(root)
        docs.append(
            {
                "id": f"{source_label}:{rel}",
                "text": text,
                "metadata": {
                    "source": source_label,
                    "filepath": str(path),
                    "filename": path.name,
                },
            }
        )
    return docs


def simple_chunk(text: str, max_chars: int = 900, overlap: int = 120) -> List[str]:
    if len(text) <= max_chars:
        return [text]

    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = start + max_chars
        chunk = text[start:end]
        chunks.append(chunk)
        if end >= len(text):
            break
        start = end - overlap
    return chunks


def chunk_documents(docs: List[Dict], max_chars: int = 900, overlap: int = 120) -> List[Dict]:
    chunked: List[Dict] = []
    for doc in docs:
        pieces = simple_chunk(doc["text"], max_chars=max_chars, overlap=overlap)
        for idx, ch in enumerate(pieces):
            chunked.append(
                {
                    "id": f"{doc['id']}#chunk{idx}",
                    "text": ch,
                    "metadata": {**doc["metadata"], "chunk_index": idx},
                }
            )
    return chunked


# ------------------------
# Embeddings
# ------------------------

_embedder = None


def embed_texts(texts: List[str]):
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedder.encode(texts, show_progress_bar=False)


# ------------------------
# Ingest and query
# ------------------------

def ingest(source: str, clear_first: bool) -> None:
    if source == "general":
        root = GENERAL_DIR
        collection = GENERAL_COLLECTION_NAME
        label = "general"
    elif source == "cs":
        root = CS_DIR
        collection = CS_COLLECTION_NAME
        label = "cs"
    else:
        raise ValueError("source must be 'general' or 'cs'")

    if not root.exists():
        print(f"[WARN] Directory {root} does not exist. Add files before ingesting.")
        return

    print(f"[INFO] Loading documents from {root}...")
    docs = load_documents(root, label)
    print(f"[INFO] Loaded {len(docs)} documents. Chunking...")

    chunked = chunk_documents(docs)
    print(f"[INFO] Created {len(chunked)} chunks. Embedding...")

    embeddings = embed_texts([c["text"] for c in chunked])
    print(f"[INFO] Embeddings ready. {'Clearing and ' if clear_first else ''}writing to '{collection}'...")

    if clear_first:
        clear_collection(collection)
    add_documents_to_collection(collection, chunked, embeddings)

    print(f"[DONE] Ingested {len(chunked)} chunks into '{collection}'.")


def query(source: str, question: str, k: int = 5):
    hits = []
    if source in ("general", "all"):
        res = query_collection(GENERAL_COLLECTION_NAME, question, k=k)
        hits.append(("general", res))
    if source in ("cs", "all"):
        res = query_collection(CS_COLLECTION_NAME, question, k=k)
        hits.append(("cs", res))

    flat = []
    for label, r in hits:
        docs = r.get("documents", [[]])[0]
        metas = r.get("metadatas", [[]])[0]
        scores = r.get("distances", [[]])[0]
        ids = r.get("ids", [[]])[0]
        for i in range(len(docs)):
            flat.append(
                {
                    "collection": label,
                    "id": ids[i],
                    "text": docs[i],
                    "metadata": metas[i],
                    "score": scores[i],
                }
            )

    if not flat:
        print("No results found.")
        return

    flat.sort(key=lambda h: h["score"])
    print(f"\nTop {min(k, len(flat))} matches for: \"{question}\"\n")
    for i, h in enumerate(flat[:k], start=1):
        meta = h["metadata"]
        print(f"=== Result {i} (collection={h['collection']}, score={h['score']:.4f}) ===")
        print(f"File: {meta.get('filename')}  [chunk {meta.get('chunk_index')}]")
        print(f"Path: {meta.get('filepath')}")
        print("-" * 80)
        print(h["text"][:1000])
        print("\n")


# ------------------------
# CLI
# ------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RAG Assistant CLI (ingest/query without the server)")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest_p = sub.add_parser("ingest", help="Ingest documents into Chroma")
    ingest_p.add_argument("--source", choices=["general", "cs"], required=True)
    ingest_p.add_argument("--clear", action="store_true", help="Clear the target collection before ingesting.")

    query_p = sub.add_parser("query", help="Query existing collections")
    query_p.add_argument("--source", choices=["general", "cs", "all"], required=True)
    query_p.add_argument("--question", required=True, type=str)
    query_p.add_argument("--k", type=int, default=5)

    return parser.parse_args()


def main():
    args = parse_args()
    if args.command == "ingest":
        ingest(args.source, clear_first=args.clear)
    elif args.command == "query":
        query(args.source, args.question, k=args.k)


if __name__ == "__main__":
    main()
