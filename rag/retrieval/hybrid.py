from rag.config import FETCH_K, TOP_K
from rag.retrieval.bm25 import bm25_search
from rag.retrieval.reranker import rerank
from rag.retrieval.semantic import search


def search_reranked(query: str, k: int = TOP_K, fetch_k: int = FETCH_K) -> list[dict]:
    return rerank(query, search(query, k=fetch_k), k)


def hybrid_search(query: str, k: int = TOP_K, fetch_k: int = FETCH_K) -> list[dict]:
    merged: dict[str, dict] = {}
    for c in search(query, k=fetch_k) + bm25_search(query, k=fetch_k):
        merged.setdefault(c["content"], c)
    return rerank(query, list(merged.values()), k)