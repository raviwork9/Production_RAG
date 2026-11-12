import os
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent

load_dotenv(ROOT / ".env", override=True)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("RAG_DATA_DIR", ROOT / "data"))
CHROMA_DIR = DATA_DIR / "chroma_db"
PDF_DIR = DATA_DIR / "pdfs"
UPLOAD_DIR = DATA_DIR / "uploads"

COLLECTION_NAME = "documents"

EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

PDF_STRATEGY = os.getenv("PDF_STRATEGY", "hi_res")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "400"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "75"))

TOP_K = int(os.getenv("TOP_K", "3"))
FETCH_K = int(os.getenv("FETCH_K", "10"))
MIN_SCORE = float(os.getenv("MIN_SCORE", "-6"))