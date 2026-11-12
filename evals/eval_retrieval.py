import os

from deepeval import evaluate
from deepeval.metrics import ContextualPrecisionMetric, ContextualRecallMetric
from deepeval.test_case import LLMTestCase

from evals.utils import (
    JUDGE_MODEL,
    GroqJudge,
    load_goldens,
    print_reasons,
    print_summary,
    summarize_by_metric,
)
from rag import config
from rag.retrieval.hybrid import hybrid_search, search_reranked
from rag.retrieval.semantic import search

GOLDEN_FILE = "retriever_goldens.json"
THRESHOLD = 0.7
RETRIEVAL_MODE = os.getenv("RETRIEVAL_MODE", "hybrid")


def retrieve(question: str) -> list[dict]:
    if RETRIEVAL_MODE == "hybrid":
        return hybrid_search(question, k=config.TOP_K, fetch_k=config.FETCH_K)
    if RETRIEVAL_MODE == "reranked":
        return search_reranked(question, k=config.TOP_K, fetch_k=config.FETCH_K)
    return search(question, k=config.TOP_K)


def run() -> dict:
    goldens = load_goldens(GOLDEN_FILE)
    judge = GroqJudge()

    test_cases = []
    for g in goldens:
        hits = retrieve(g["question"])
        test_cases.append(
            LLMTestCase(
                name=g["id"],
                input=g["question"],
                expected_output=g["ground_truth"],
                retrieval_context=[h["content"] for h in hits],
                actual_output="(generator not evaluated in this run)",
            )
        )

    metrics = [
        ContextualRecallMetric(threshold=THRESHOLD, model=judge, include_reason=True, async_mode=False),
        ContextualPrecisionMetric(threshold=THRESHOLD, model=judge, include_reason=True, async_mode=False),
    ]

    uses_reranker = RETRIEVAL_MODE != "semantic"
    result = evaluate(
        test_cases=test_cases,
        metrics=metrics,
        hyperparameters={
            "retriever": RETRIEVAL_MODE,
            "reranker_model": config.RERANK_MODEL if uses_reranker else None,
            "fetch_k": config.FETCH_K if uses_reranker else None,
            "embedding_model": config.EMBED_MODEL,
            "chunk_size": config.CHUNK_SIZE,
            "chunk_overlap": config.CHUNK_OVERLAP,
            "top_k": config.TOP_K,
            "judge_model": JUDGE_MODEL,
            "golden_set": GOLDEN_FILE,
        },
    )
    print_reasons(result)
    return summarize_by_metric(result)


if __name__ == "__main__":
    print_summary(f"retrieval ({RETRIEVAL_MODE})", run())