import os
from functools import lru_cache

from dotenv import load_dotenv
from openai import OpenAI

from rag.config import FETCH_K, MIN_SCORE, TOP_K
from rag.generation.prompts import SYSTEM_PROMPT, USER_TEMPLATE
from rag.guardrails.guardrails import OUT_OF_CONTEXT, guard_input, guard_output
from rag.retrieval.hybrid import hybrid_search

load_dotenv()

GEN_MODEL = os.getenv("GEN_MODEL", "openai/gpt-oss-20b")
MAX_TOKENS = int(os.getenv("GEN_MAX_TOKENS", "1000"))


@lru_cache(maxsize=1)
def get_client() -> OpenAI:
    return OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=os.environ["GROQ_API_KEY"].strip().strip('"').strip("'"),
    )


def build_context(hits: list[dict]) -> str:
    return "\n\n".join(f"[p.{h['page']}] {h['content']}" for h in hits)


def blocked(message: str, reason: str) -> dict:
    return {"answer": message, "contexts": [], "sources": [], "guard": reason}


def generate_answer(question: str, k: int = TOP_K, fetch_k: int = FETCH_K) -> dict:
    clean_question, refusal, reason = guard_input(question)
    if refusal:
        return blocked(refusal, f"input:{reason}")

    hits = hybrid_search(clean_question, k=fetch_k, fetch_k=fetch_k)
    hits = [h for h in hits if h.get("rerank_score", 0) > MIN_SCORE][:k]
    if not hits:
        return blocked(OUT_OF_CONTEXT, "retrieval:no_relevant_chunks")

    user_prompt = USER_TEMPLATE.format(
        context=build_context(hits), question=clean_question
    )

    out = get_client().chat.completions.create(
        model=GEN_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
        max_tokens=MAX_TOKENS,
    )
    answer, out_reason = guard_output((out.choices[0].message.content or "").strip())

    return {
        "answer": answer,
        "contexts": [h["content"] for h in hits],
        "sources": [
            {"source": h["source"], "page": h["page"], "score": h.get("rerank_score")}
            for h in hits
        ],
        "guard": f"output:{out_reason}" if out_reason else None,
    }