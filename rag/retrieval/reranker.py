from functools import lru_cache

from rag.config import RERANK_MODEL


@lru_cache(maxsize=1)
def get_reranker():
    from sentence_transformers import CrossEncoder

    return CrossEncoder(RERANK_MODEL)


def rerank(query: str, candidates: list[dict], k: int) -> list[dict]:
    if not candidates:
        return []
    scores = get_reranker().predict([(query, c["content"]) for c in candidates])
    for c, s in zip(candidates, scores):
        c["rerank_score"] = float(s)
    candidates.sort(key=lambda c: c["rerank_score"], reverse=True)
    return candidates[:k]