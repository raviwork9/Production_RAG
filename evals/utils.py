import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from deepeval.models import DeepEvalBaseLLM

load_dotenv()

GOLDENS_DIR = Path(__file__).resolve().parent / "goldens"
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "llama-3.3-70b-versatile")


class GroqJudge(DeepEvalBaseLLM):
    def __init__(self, model_name: str = JUDGE_MODEL):
        self.model_name = model_name
        self.client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=os.environ["GROQ_API_KEY"].strip().strip('"').strip("'"),
        )

    def load_model(self):
        return self.client

    def generate(self, prompt: str, schema=None):
        if schema is not None:
            prompt += "\n\nRespond with valid JSON only. No markdown, no extra text."

        for attempt in range(6):
            kwargs = {}
            if schema is not None and attempt < 2:
                kwargs["response_format"] = {"type": "json_object"}

            try:
                out = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=4000,
                    **kwargs,
                )
                text = (out.choices[0].message.content or "").strip()
                if schema is None:
                    return text
                data = json.loads(text[text.find("{"): text.rfind("}") + 1])
                return schema(**data)
            except Exception as e:
                msg = str(e)
                retryable = "429" in msg or "JSON" in msg or isinstance(e, ValueError)
                if not retryable or attempt == 5:
                    raise
                wait = 10 * (attempt + 1) if "429" in msg else 2
                print(f"Judge error, retrying in {wait}s...")
                time.sleep(wait)

    async def a_generate(self, prompt: str, schema=None):
        return self.generate(prompt, schema)

    def get_model_name(self) -> str:
        return self.model_name


def load_goldens(name: str) -> list[dict]:
    path = GOLDENS_DIR / name
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def summarize_by_metric(result) -> dict:
    scores: dict[str, list] = {}
    for tr in result.test_results:
        for m in tr.metrics_data:
            scores.setdefault(m.name, []).append((m.score, m.success))
    return {
        name: {
            "avg_score": sum(s for s, _ in vals) / len(vals),
            "pass_rate": sum(1 for _, ok in vals if ok) / len(vals),
            "cases": len(vals),
        }
        for name, vals in scores.items()
    }


def print_summary(title: str, summary: dict) -> None:
    print(f"\n=== {title} summary ===")
    for name, s in summary.items():
        print(f"{name}: avg={s['avg_score']:.2f} pass_rate={s['pass_rate']:.0%} ({s['cases']} cases)")


def print_reasons(result) -> None:
    print("\n=== per-case results ===")
    for tr in result.test_results:
        for m in tr.metrics_data:
            status = "PASS" if m.success else "FAIL"
            print(f"\n{tr.name} | {m.name} | {m.score:.2f} | {status}")
            print(f"Reason: {m.reason}")