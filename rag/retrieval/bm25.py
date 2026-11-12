import re

from rank_bm25 import BM25Okapi

from rag.retrieval.store import get_collection

_index = None
_docs: list[dict] = []
_built_count = -1


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def _build() -> None:
    global _index, _docs, _built_count
    data = get_collection().get(include=["documents", "metadatas"])
    _docs = [
        {"content": d, "page": m["page"], "source": m["source"]}
        for d, m in zip(data["documents"], data["metadatas"])
    ]
    _index = BM25Okapi([tokenize(d["content"]) for d in _docs]) if _docs else None
    _built_count = len(_docs)


def bm25_search(query: str, k: int = 5) -> list[dict]:
    if get_collection().count() != _built_count:
        _build()
    if _index is None:
        return []

    scores = _index.get_scores(tokenize(query))
    top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [dict(_docs[i]) for i in top if scores[i] > 0]