from rag.embeddings import embed_texts
from rag.retrieval.store import get_collection


def search(query: str, k: int = 3) -> list[dict]:
    collection = get_collection()
    n = min(k, collection.count())
    if n == 0:
        return []

    res = collection.query(query_embeddings=embed_texts([query]), n_results=n)
    return [
        {"content": d, "page": m["page"], "source": m["source"], "distance": dist}
        for d, m, dist in zip(
            res["documents"][0], res["metadatas"][0], res["distances"][0]
        )
    ]