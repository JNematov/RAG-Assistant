"""
Ingest local text/markdown files into Chroma collections.

Usage examples:
  python tools/ingest_cly.py --target cs --reset
  python tools/ingest_cly.py --target general
  python tools/ingest_cly.py --target all --chunk-size 900 --overlap 150
"""

import argparse
import sys
from pathlib import Path
from typing import Iterable, List, Dict, Tuple

# Ensure project root is on sys.path so `backend` imports resolve when run as a script.
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from sentence_transformers import SentenceTransformer

from backend.core.rag.config import CS_DIR, GENERAL_DIR, CS_COLLECTION_NAME, GENERAL_COLLECTION_NAME
from backend.core.rag.vector_store import add_documents_to_collection, clear_collection


def load_text_files(root: Path, patterns: Iterable[str]) -> List[Path]:
    files: List[Path] = []
    for pat in patterns:
        files.extend(root.rglob(pat))
    return sorted({p for p in files if p.is_file()})


def load_text_from_file(path: Path) -> Tuple[str, str]:
    """
    Return (text, filetype). Supports txt/md and pdf (best-effort).
    """
    suffix = path.suffix.lower()
    if suffix in (".txt", ".md"):
        return path.read_text(encoding="utf-8", errors="ignore"), "text"

    if suffix == ".pdf":
        try:
            from pypdf import PdfReader  # lazy import
        except ImportError:
            raise RuntimeError("pypdf is required to ingest PDFs. Install with: pip install pypdf")

        reader = PdfReader(str(path))
        pages = []
        for page in reader.pages:
            pages.append(page.extract_text() or "")
        return "\n".join(pages), "pdf"

    return "", "unknown"


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """
    Simple character-based chunking with overlap.
    """
    chunks: List[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(n, start + chunk_size)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap
        if start <= 0:
            start = end
    return chunks


def build_chunk_docs(file_path: Path, root_dir: Path, source_label: str, chunk_size: int, overlap: int) -> List[Dict]:
    """
    Build chunk-level docs for a single file.
    """
    raw_text, ftype = load_text_from_file(file_path)
    if not raw_text.strip():
        return []

    try:
        relative = file_path.relative_to(root_dir)
    except ValueError:
        relative = file_path.name

    chunks = chunk_text(raw_text, chunk_size, overlap)
    docs: List[Dict] = []
    for idx, chunk in enumerate(chunks):
        doc_id = f"{source_label}:{relative}:chunk{idx}"
        docs.append(
            {
                "id": doc_id,
                "text": chunk,
                "metadata": {
                    "source": source_label,
                    "file": file_path.name,
                    "path": str(file_path),
                    "chunk_index": idx,
                    "filetype": ftype,
                },
            }
        )
    return docs


def ingest_collection(
    collection_name: str,
    source_label: str,
    root_dir: Path,
    model: SentenceTransformer,
    *,
    chunk_size: int,
    overlap: int,
    reset: bool,
    batch_size: int,
) -> None:
    if reset:
        print(f"[info] Clearing collection '{collection_name}'...")
        clear_collection(collection_name)

    patterns = ("*.txt", "*.md", "*.pdf")
    files = load_text_files(root_dir, patterns)
    if not files:
        print(f"[warn] No files found under {root_dir}")
        return

    total_chunks = 0
    for file_path in files:
        docs = build_chunk_docs(file_path, root_dir, source_label, chunk_size, overlap)
        if not docs:
            continue

        print(f"[info] {file_path} -> {len(docs)} chunk(s)")
        texts = [d["text"] for d in docs]
        embeddings = model.encode(texts, show_progress_bar=False, batch_size=batch_size)

        add_documents_to_collection(collection_name, docs, embeddings)
        total_chunks += len(docs)

    print(f"[done] Ingested {total_chunks} chunk(s) into '{collection_name}'.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest notes into Chroma.")
    parser.add_argument(
        "--target",
        choices=["cs", "general", "all"],
        default="all",
        help="Which collection(s) to ingest.",
    )
    parser.add_argument("--chunk-size", type=int, default=800, help="Chunk size (characters).")
    parser.add_argument("--overlap", type=int, default=120, help="Overlap between chunks (characters).")
    parser.add_argument("--reset", action="store_true", help="Clear the target collection(s) first.")
    parser.add_argument("--batch-size", type=int, default=64, help="Embedding batch size to control memory use.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    if args.target in ("cs", "all"):
        ingest_collection(
            CS_COLLECTION_NAME,
            "cs",
            CS_DIR,
            model,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
            reset=args.reset,
            batch_size=args.batch_size,
        )

    if args.target in ("general", "all"):
        ingest_collection(
            GENERAL_COLLECTION_NAME,
            "general",
            GENERAL_DIR,
            model,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
            reset=args.reset,
            batch_size=args.batch_size,
        )


if __name__ == "__main__":
    main()
