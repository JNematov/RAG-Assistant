from typing import Literal, List, Dict
from .config import GENERAL_COLLECTION_NAME, CS_COLLECTION_NAME
from .vector_store import query_collection


SourceType = Literal["general", "cs", "all"]


def query_sources(source: SourceType, question: str, k: int = 5) -> List[Dict]:
    results = []

    if source in ("general", "all"):
        res_gen = query_collection(GENERAL_COLLECTION_NAME, question, k=k)
        results.append(("general", res_gen))

    if source in ("cs", "all"):
        res_cs = query_collection(CS_COLLECTION_NAME, question, k=k)
        results.append(("cs", res_cs))

    # Flatten & normalize
    hits: List[Dict] = []
    for label, r in results:
        docs = r.get("documents", [[]])[0]
        metas = r.get("metadatas", [[]])[0]
        scores = r.get("distances", [[]])[0]  # chroma uses distance; lower is better
        ids = r.get("ids", [[]])[0]

        for i in range(len(docs)):
            hits.append({
                "collection": label,
                "id": ids[i],
                "text": docs[i],
                "metadata": metas[i],
                "score": scores[i],
            })

    # Sort by score ascending (closer = better)
    hits.sort(key=lambda h: h["score"])
    return hits