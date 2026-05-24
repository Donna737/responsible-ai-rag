# responsible-ai-rag
# Responsible AI Research Assistant
### A production-grade RAG pipeline for querying academic research on Responsible AI

> **Status:** Pipeline complete  · Streamlit app in progress  · Docker + CI/CD + deployment coming 

---

## What This Is

A Retrieval-Augmented Generation (RAG) system that lets you ask research questions and get grounded, cited answers from a curated corpus of 20 academic papers on Responsible AI.

This is not a wrapper around a chatbot. The system retrieves specific passages from real papers, reranks them for relevance, and instructs the LLM to answer *only* from what the papers say — with inline citations traceable to the source. If the answer isn't in the corpus, it says so.

Built as a portfolio project to demonstrate end-to-end AI engineering — from data collection and document processing through embedding, vector search, reranking, and grounded generation — with production practices including modular design, persistent storage, scalable ingestion, and (coming) Docker and CI/CD.

---

## Architecture

```
                        ┌─────────────────────────────────┐
                        │         INGESTION (once)         │
                        │                                  │
  arXiv PDFs ──► PyMuPDF extract ──► Clean & chunk ──► Embed (MiniLM) ──► ChromaDB
                                                                               │
                        ┌─────────────────────────────────┐                   │
                        │         QUERY (per request)      │                   │
                        │                                  │                   │
  User question ──► Embed ──► Retrieve top 20 ────────────┘
                                     │
                              Rerank (CrossEncoder)
                                     │
                              Top 5 diverse chunks
                                     │
                         LLM (Llama 3.1 via Groq)
                                     │
                        Answer + inline citations
```

**Two-stage retrieval** is the core design decision: fast approximate search (bi-encoder embeddings) narrows 2,874 chunks to 20 candidates, then a slower but more accurate cross-encoder reranker scores each (question, chunk) pair together to find the true top 5. This pattern balances speed and quality — the same approach used in production search systems.

---

## Stack

| Component | Tool | Why |
|---|---|---|
| PDF extraction | PyMuPDF | Fast, handles complex layouts |
| Chunking | LangChain RecursiveCharacterTextSplitter | Natural boundary-aware splitting |
| Embeddings | `all-MiniLM-L6-v2` | Fast, free, strong semantic search |
| Vector store | ChromaDB (persistent) | Simple, local, production-ready API |
| Reranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Accurate pair-level relevance scoring |
| LLM | Llama 3.1 8B via Groq | Fast inference, free tier |
| UI | Streamlit (in progress) | Clean, deployable, no frontend overhead |

---

## Corpus

20 research papers on Responsible AI from arXiv, covering 2020–2024.

Curated with a deliberate split:
- **10 most cited papers** — foundational works establishing the field's core concepts
- **10 most recent papers** — 2022–2024 work the LLM hasn't seen, where RAG adds the most value

Topics covered: fairness, transparency, accountability, explainability, trustworthy AI, bias mitigation, AI governance, EU AI Act context, power dynamics in AI development.

Total: **1949 chunks** · avg chunk size **847 characters**

---

## Example Output

**Q: What is the relationship between explainability and accountability in AI?**

```
Retrieved 20 chunks → Reranked to 5 chunks

Sources used:
  1] Connecting the Dots in Trustworthy Artificial Intelligence (rerank score: 5.370)
  [2] Connecting the Dots in Trustworthy Artificial Intelligence (rerank score: 4.527)
  [3] Explainable Artificial Intelligence (XAI): Concepts, Taxonomies,
    Opportunities and Challenges toward Responsible AI  (rerank score: 1.996)
  [4] Harnessing Metacognition for Safe and Responsible AI  (rerank score: 1.747)
  [5] Towards Responsible AI for Education: Hybrid Human-AI to Confront
    the Elephant in the Room  (rerank score: 1.744)
```

> According to the provided research papers, explainability and accountability are related but distinct concepts. Explainability refers to the ability of an AI system to provide clear and understandable reasons for its functioning and decision-making processes [Source 3]. Accountability, on the other hand, is a matter of compliance with ethical and legal standards, answerability, reporting, and oversight, and attribution and enforcement of consequences [Source 2].
>
> While explainability can contribute to accountability by providing insights into decision-making processes, it is not a direct guarantee of accountability. Accountability requires a broader set of measures, including compliance with regulations and the ability to distribute costs, risks, and liabilities among stakeholders [Source 2].
>
> In other words, explainability is a necessary but not sufficient condition for accountability. An AI system can be explainable but still lack accountability if it does not comply with relevant regulations or provide adequate oversight mechanisms [Source 2].

---

## Key Design Decisions

**Why RAG and not fine-tuning?**
The corpus is small (20 papers), changes over time, and answers need to be traceable to specific sources. RAG is the right tool — fine-tuning would bake knowledge in without citation ability and would need retraining whenever new papers are added.

**Why two-stage retrieval?**
Running a cross-encoder on all 2,874 chunks per query would be too slow. Bi-encoder embeddings do a fast broad search first, then the cross-encoder does accurate pair-level scoring on only the top 20 candidates.

**Why diversity enforcement?**
Without it, one highly relevant paper dominates all top-K results. Capping at 2 chunks per paper forces the system to surface perspectives from multiple sources — more useful and more honest.

**Scalable ingestion**
The ingestion script checks existing chunk IDs in ChromaDB before embedding. Adding new papers means dropping PDFs in the folder and rerunning — only the new chunks get embedded.

---

## Known Limitations

- **Fixed-size chunking** can cut mid-sentence on long complex sentences. `SemanticChunker` (LangChain) would be a proper fix — splits on topic boundaries rather than character count.
- **Topic drift** on out-of-scope questions — the system retrieves the closest chunks even when a question has no relevant answer in the corpus. A confidence threshold filter is planned.
- **PDF extraction quality** varies — scanned or image-based PDFs extract poorly. OCR integration (Tesseract) would handle these.

---

## Roadmap

- [x] Data collection — arXiv API + manual curation
- [x] PDF extraction and cleaning
- [x] Chunking with overlap
- [x] Embedding with sentence-transformers
- [x] Persistent vector store (ChromaDB)
- [x] Two-stage retrieval with reranking
- [x] Diversity enforcement
- [x] Grounded generation with citations (Groq / Llama 3.1)
- [x] Stress testing — hallucination, out-of-scope, conflicting info
- [ ] Streamlit UI with streaming responses
- [ ] Docker containerization
- [ ] GitHub Actions CI/CD
- [ ] Deploy on Hugging Face Spaces

---


