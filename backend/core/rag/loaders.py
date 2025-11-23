from pathlib import Path
from typing import List, Dict
import textwrap

from pypdf import PdfReader


def load_txt_or_md(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def load_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    texts = []
    for page in reader.pages:
        texts.append(page.extract_text() or "")
    return "\n".join(texts)


def load_documents_from_dir(root: Path, source_label: str) -> List[Dict]:
    """
    Returns list of dicts: { 'id', 'text', 'metadata' }
    """
    docs = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue

        if path.suffix.lower() in [".txt", ".md"]:
            raw = load_txt_or_md(path)
        elif path.suffix.lower() == ".pdf":
            raw = load_pdf(path)
        else:
            continue  # skip other types for now

        text = textwrap.dedent(raw).strip()
        if not text:
            continue

        docs.append({
            "id": f"{source_label}:{path.relative_to(root)}",
            "text": text,
            "metadata": {
                "source": source_label,
                "filepath": str(path),
                "filename": path.name,
            }
        })
    return docs