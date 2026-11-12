from rag.config import PDF_DIR
from rag.generation.generator import generate_answer
from rag.ingestion.pipeline import ingest_all
from rag.logging_setup import setup_logging
from rag.retrieval.store import count


def chat() -> None:
    if count() == 0:
        print(f"The database is empty. Put a PDF in {PDF_DIR} and run again.")
        return

    print(f"\nAsk about your PDFs ({count()} chunks stored). Type 'exit' to quit.")
    while True:
        question = input("\n> ").strip()
        if question.lower() in {"exit", "quit", ""}:
            break

        result = generate_answer(question)
        print("\n" + result["answer"])

        if result.get("guard"):
            print(f"\n[guard: {result['guard']}]")
        if result["sources"]:
            pages = sorted({(s["source"], s["page"]) for s in result["sources"]})
            print("\nSources:", ", ".join(f"{src} p.{pg}" for src, pg in pages))


def main() -> None:
    setup_logging()

    PDF_DIR.mkdir(parents=True, exist_ok=True)
    pdfs = sorted(PDF_DIR.glob("*.pdf"))
    if pdfs:
        ingest_all(pdfs)
    else:
        print(f"No PDFs found in {PDF_DIR}")

    chat()


if __name__ == "__main__":
    main()