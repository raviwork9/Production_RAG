from functools import lru_cache

import chromadb

from rag.config import CHROMA_DIR, COLLECTION_NAME
from rag.embeddings import embed_texts


@lru_cache(maxsize=1)
def get_collection():
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def already_stored(source: str) -> bool:
    res = get_collection().get(where={"source": source}, limit=1)
    return len(res["ids"]) > 0


def delete_source(source: str) -> None:
    get_collection().delete(where={"source": source})


def add_chunks(chunks: list[dict], source: str) -> int:
    if not chunks:
        return 0
    texts = [c["content"] for c in chunks]
    get_collection().add(
        ids=[f"{source}-{i}" for i in range(len(chunks))],
        documents=texts,
        embeddings=embed_texts(texts),
        metadatas=[{"source": source, "page": c["page"] or 0} for c in chunks],
    )
    return len(chunks)


def list_sources() -> list[str]:
    data = get_collection().get(include=["metadatas"])
    return sorted({m["source"] for m in data["metadatas"]})


def count() -> int:
    return get_collection().count()