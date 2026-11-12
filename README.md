# Production RAG: Document Question Answering System

A Retrieval-Augmented Generation (RAG) system that answers questions from long documents (100+ pages) with context-grounded responses. It combines hybrid retrieval, cross-encoder reranking, automated evaluation and Responsible AI guardrails.

## Features

- **Hybrid search:** dense vector retrieval plus BM25 keyword search, so it handles both semantic and exact-term queries.
- **Cross-encoder reranking:** re-scores the retrieved candidates so the most relevant context reaches the LLM.
- **Grounded answers:** responses are generated strictly from the retrieved document context.
- **Responsible AI guardrails:** protection at the input, retrieval and output layers, including PII masking and prompt injection defense.
- **Automated evaluation:** quality is measured with RAGAS (recall, precision and faithfulness).

## Results

Measured with RAGAS on the evaluation set:

| Metric | Score |
|---|---|
| Recall | ≥ 0.96 |
| Precision | ≥ 0.92 |
| Faithfulness | ≥ 0.98 |
| Prompt injection neutralization (testing) | ≥ 99% |

## Architecture

```
User query
   │
   ▼
Input guardrails (PII masking, prompt injection defense)
   │
   ▼
Hybrid retrieval (dense vectors + BM25)
   │
   ▼
Cross-encoder reranking
   │
   ▼
Retrieval guardrails
   │
   ▼
LLM answer generation (grounded in retrieved context)
   │
   ▼
Output guardrails
   │
   ▼
Final answer
```

## Project Structure

```
Production_RAG/
├── app/            # Application layer (interface / entry point)
├── rag/            # Core RAG pipeline: ingestion, retrieval, reranking, generation
├── evals/          # RAGAS evaluation code and datasets
├── scripts/        # Helper scripts (e.g. indexing documents)
├── requirements.txt
└── .gitignore
```


**Ravi Kumar**
GitHub: [@raviwork9](https://github.com/raviwork9)
Email: raviworkingat9@gmail.com
