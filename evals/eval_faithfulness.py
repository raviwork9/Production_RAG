from deepeval import evaluate
from deepeval.metrics import FaithfulnessMetric
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
from rag.generation.generator import GEN_MODEL, generate_answer

GOLDEN_FILE = "retriever_goldens.json"
THRESHOLD = 0.7


def run() -> dict:
    goldens = load_goldens(GOLDEN_FILE)
    judge = GroqJudge()

    test_cases = []
    for g in goldens:
        answer = generate_answer(g["question"])
        print(f"{g['id']}: {answer['answer'][:100]!r}")
        test_cases.append(
            LLMTestCase(
                name=g["id"],
                input=g["question"],
                actual_output=answer["answer"],
                retrieval_context=answer["contexts"],
            )
        )

    metrics = [
        FaithfulnessMetric(threshold=THRESHOLD, model=judge, include_reason=True, async_mode=False),
    ]

    result = evaluate(
        test_cases=test_cases,
        metrics=metrics,
        hyperparameters={
            "generator_model": GEN_MODEL,
            "top_k": config.TOP_K,
            "min_score": config.MIN_SCORE,
            "judge_model": JUDGE_MODEL,
            "golden_set": GOLDEN_FILE,
        },
    )
    print_reasons(result)
    return summarize_by_metric(result)


if __name__ == "__main__":
    print_summary("faithfulness", run())