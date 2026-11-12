from functools import lru_cache

from langchain_text_splitters import RecursiveCharacterTextSplitter
from unstructured.partition.pdf import partition_pdf

from rag.config import CHUNK_OVERLAP, CHUNK_SIZE, PDF_STRATEGY


@lru_cache(maxsize=1)
def get_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def load_pdf(path: str) -> list[dict]:
    elements = partition_pdf(filename=path, strategy=PDF_STRATEGY)

    pages: dict[int, list[str]] = {}
    for el in elements:
        if el.text and el.text.strip():
            pages.setdefault(el.metadata.page_number, []).append(el.text.strip())

    splitter = get_splitter()
    results = []
    for page, parts in pages.items():
        for piece in splitter.split_text("\n\n".join(parts)):
            if piece.strip():
                results.append({"content": piece, "page": page})
    return results