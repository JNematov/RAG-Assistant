from pathlib import Path

# Go up TWO levels from rag/ to project root:
# rag/ -> core/ -> backend/ -> project root (we need parents[3])
BASE_DIR = Path(__file__).resolve().parents[3]  # project root

DATA_DIR = BASE_DIR / "data"
GENERAL_DIR = DATA_DIR / "general"
CS_DIR = DATA_DIR / "cs_notes"

CHROMA_DIR = BASE_DIR / "db"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Collections
GENERAL_COLLECTION_NAME = "general_docs"
CS_COLLECTION_NAME = "cs_docs"