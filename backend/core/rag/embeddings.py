from sentence_transformers import SentenceTransformer
from typing import List
from .config import EMBEDDING_MODEL_NAME

_model = None

def get_embedding_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model


def embed_texts(texts: List[str]):
    model = get_embedding_model()
    return model.encode(texts, show_progress_bar=False)