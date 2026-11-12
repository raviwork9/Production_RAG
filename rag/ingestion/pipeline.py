import logging
from pathlib import Path

from rag.ingestion.loader import load_pdf
from rag.retrieval.store import add_chunks, already_stored, delete_source

log = logging.getLogger(__name__)


def ingest_file(pdf_path: str | Path, force: bool = False) -> int:
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(path)

    source = path.name
    if already_stored(source):
        if not force:
            log.info("%s already stored, skipping chunking and embedding", source)
            return 0
        delete_source(source)

    log.info("%s: parsing and chunking", source)
    chunks = load_pdf(str(path))
    if not chunks:
        log.warning("%s: no text found, nothing stored", source)
        return 0

    stored = add_chunks(chunks, source)
    log.info("%s: stored %d chunks", source, stored)
    return stored


def ingest_all(pdf_paths: list[str | Path], force: bool = False) -> dict[str, int]:
    results = {}
    for p in pdf_paths:
        try:
            results[Path(p).name] = ingest_file(p, force=force)
        except FileNotFoundError:
            log.warning("File not found, skipping: %s", p)
    return results