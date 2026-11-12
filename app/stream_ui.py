import logging
from pathlib import Path

import streamlit as st

from rag.config import PDF_DIR, UPLOAD_DIR
from rag.generation.generator import generate_answer
from rag.ingestion.pipeline import ingest_all, ingest_file
from rag.logging_setup import setup_logging
from rag.retrieval.store import already_stored, count, list_sources

MAX_UPLOAD_MB = 20

setup_logging()
log = logging.getLogger(__name__)

st.set_page_config(page_title="PDF Chatbot", page_icon="📄", layout="wide")
st.title("📄 PDF Chatbot")


@st.cache_resource(show_spinner="Checking PDFs and indexing new ones...")
def prepare() -> bool:
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    ingest_all(sorted(PDF_DIR.glob("*.pdf")))
    return True


def render_chunks(chunks: list[dict]) -> None:
    for i, c in enumerate(chunks, start=1):
        score = f" | score {c['score']:.2f}" if c.get("score") is not None else ""
        st.markdown(f"**Chunk {i}** — {c['source']}, page {c['page']}{score}")
        st.info(c["text"])


def handle_upload(uploaded) -> None:
    name = Path(uploaded.name).name

    if uploaded.size > MAX_UPLOAD_MB * 1024 * 1024:
        st.error(f"{name} is larger than {MAX_UPLOAD_MB} MB.")
        return
    if already_stored(name):
        st.info(f"{name} is already stored, skipping embedding")
        return

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    save_path = UPLOAD_DIR / name
    save_path.write_bytes(uploaded.getbuffer())

    try:
        with st.spinner(f"Chunking and embedding {name}..."):
            stored = ingest_file(save_path)
    except Exception:
        log.exception("Failed to ingest %s", name)
        st.error(f"Could not process {name}. Check that it is a text-based PDF.")
        return

    if stored:
        st.success(f"{name}: {stored} chunks stored")
    else:
        st.warning(f"No text found in {name}, nothing was stored.")


prepare()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "processed_uploads" not in st.session_state:
    st.session_state.processed_uploads = set()

with st.sidebar:
    st.header("Knowledge base")

    uploaded = st.file_uploader("Upload a PDF", type="pdf")
    if uploaded is not None and uploaded.name not in st.session_state.processed_uploads:
        handle_upload(uploaded)
        st.session_state.processed_uploads.add(uploaded.name)

    st.write(f"Chunks stored: **{count()}**")
    st.write("Files:")
    for name in list_sources():
        st.markdown(f"- {name}")

    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("guard"):
            st.caption(f"guard: {m['guard']}")
        if m.get("chunks"):
            with st.expander("Retrieved chunks", expanded=True):
                render_chunks(m["chunks"])

question = st.chat_input("Ask a question about your PDFs")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        if count() == 0:
            st.warning("No PDFs stored yet. Upload one in the sidebar.")
            st.stop()

        with st.spinner("Thinking..."):
            try:
                result = generate_answer(question)
            except Exception:
                log.exception("generate_answer failed")
                st.error("Something went wrong while answering. Please try again.")
                st.stop()

        st.markdown(result["answer"])
        if result.get("guard"):
            st.caption(f"guard: {result['guard']}")

        chunks = [
            {"source": s["source"], "page": s["page"], "score": s.get("score"), "text": c}
            for s, c in zip(result["sources"], result["contexts"])
        ]
        if chunks:
            with st.expander("Retrieved chunks", expanded=True):
                render_chunks(chunks)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result["answer"],
            "chunks": chunks,
            "guard": result.get("guard"),
        }
    )