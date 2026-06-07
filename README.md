---
title: Responsible AI Research Assistant
emoji: 🔍
sdk: docker
app_file: app.py
---

# Responsible AI Research Assistant
### A production-grade RAG pipeline for querying academic research on Responsible AI

> **Live demo:** [huggingface.co/spaces/DonaArabi99/responsible-ai-rag](https://huggingface.co/spaces/DonaArabi99/responsible-ai-rag)

---

## What This Is

A Retrieval-Augmented Generation (RAG) system that lets you ask research questions and get grounded, cited answers from a curated corpus of 20 academic papers on Responsible AI.

This is not a wrapper around a chatbot. The system retrieves specific passages from real papers, reranks them for relevance, and instructs the LLM to answer *only* from what the papers say — with inline citations traceable to the source. If the answer isn't in the corpus, it says so.

Built as a portfolio project to demonstrate end-to-end AI engineering — from data collection and document processing through embedding, vector search, reranking, and grounded generation — with production practices including modular design, persistent storage, scalable ingestion, Docker containerization, and Hugging Face deployment.

---

## Architecture

```
                        ┌─────────────────────────────────┐
                        │         INGESTION (once)         │
                        │                                  │
  arXiv PDFs ──► pymupdf4llm extract ──► Clean & chunk ──► Embed (MiniLM) ──► ChromaDB
                                                                                   │
                        ┌─────────────────────────────────┐                       │
                        │         QUERY (per request)      │                       │
                        │                                  │                       │
  User question ──► Embed ──► Retrieve top 20 ────────────┘
                                     │
                              Rerank (CrossEncoder)
                                     │
                              Top 5 diverse chunks
                              (max 2 per paper)
                                     │
                         LLM (Llama 3.1 8B via Groq)
                                     │
                        Answer + inline citations
```

**Two-stage retrieval** is the core design decision: fast approximate search (bi-encoder embeddings) narrows ~2,000 chunks to 20 candidates, then a slower but more accurate cross-encoder reranker scores each (question, chunk) pair together to find the true top 5. This pattern balances speed and quality — the same approach used in production search systems.

---

## Stack

| Component | Tool | Why |
|---|---|---|
| PDF extraction | pymupdf4llm | Clean markdown output, handles complex layouts |
| Chunking | LangChain RecursiveCharacterTextSplitter | Natural boundary-aware splitting |
| Embeddings | `all-MiniLM-L6-v2` | Fast, free, strong semantic search |
| Vector store | ChromaDB (persistent) | Simple, local, production-ready API |
| Reranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Accurate pair-level relevance scoring |
| LLM | Llama 3.1 8B via Groq | Fast inference, free tier |
| UI | Streamlit | Clean, deployable, streaming responses |
| Deployment | Hugging Face Spaces (Docker) | Free, public, AI community standard |

---

## Corpus

20 research papers on Responsible AI from arXiv, covering 2020–2024.

Curated with a deliberate split:
- **10 most cited papers** — foundational works establishing the field's core concepts
- **10 most recent papers** — 2022–2024 work the LLM hasn't seen, where RAG adds the most value

Topics covered: fairness, transparency, accountability, explainability, trustworthy AI, bias mitigation, AI governance, EU AI Act context, power dynamics in AI development.

> > Papers are not included in this repository for copyright reasons. See [`papers/README.md`](papers/README.md) for the full list of papers and where to download them.

---

## Example Output

> **Note:** Answers vary slightly between runs due to LLM temperature sampling. The retrieved sources and core content remain consistent — only the phrasing differs.

**Q: How does the opacity of deep learning models undermine accountability in responsible AI frameworks?**

```
Retrieved 20 chunks → Reranked to 5 chunks

Sources used:
  [1] Rethinking responsible AI from ethical pillars to sociotechnical practice
      Similarity: 0.629 · score 4.881
  [2] Responsible AI in Marketing: AI Booing and AI Washing Cycle of AI Mistrust
      Similarity: 0.612 · score 3.562
  [3] Rethinking responsible AI from ethical pillars to sociotechnical practice
      Similarity: 0.645 · score 3.433
  [4] Explainable Artificial Intelligence (XAI): Concepts, Taxonomies,
      Opportunities and Challenges toward Responsible AI
      Similarity: 0.680 · score 1.776
  [5] Harnessing Metacognition for Safe and Responsible AI
      Similarity: 0.648 · score 1.554
```

> The opacity of deep learning models significantly undermines accountability in responsible AI frameworks by obscuring causal relationships between design choices and outcomes [Source 1]. This opacity, also known as the "black box problem," arises from the complex decision-making processes of deep learning models, which lack transparency [Source 2]. As a result, it becomes extraordinarily difficult to establish clear causal links between specific human decisions and specific AI behaviours, exacerbating accountability gaps [Source 1].


>
> Furthermore, the opacity of deep learning models complicates interpretability and accountability, making it challenging for stakeholders to understand AI-driven conclusions [Source 5]. This lack of transparency undermines the principles of fairness, accountability, and value alignment, which are essential components of responsible AI frameworks [Source 3]. In fact, the guidelines behind Responsible AI emphasize the importance of explainability, fairness, accountability, and privacy when implementing AI models in real environments [Source 4].


>
> The consequences of this opacity are far-reaching, as unexplained AI-generated recommendations can undermine consumer trust and ethical accountability in marketing operations [Source 2]. Moreover, the over-reliance on AI-generated insights can lead to a disregard for contextual factors or human expertise, further exacerbating accountability gaps [Source 2].
>
> To address these challenges, researchers have identified research needs around the concepts and metrics to evaluate the explainability of ML models, and outlined research directions toward making Deep Learning models more understandable [Source 4]. Additionally, the concept of Responsible Artificial Intelligence imposes the systematic adoption of several AI principles, including explainability, fairness, accountability, and privacy, to ensure that AI models are of practical use in real environments [Source 4].
>
> In conclusion, the opacity of deep learning models undermines accountability in responsible AI frameworks by obscuring causal relationships between design choices and outcomes, complicating interpretability and accountability, and undermining the principles of fairness, accountability, and value alignment. Addressing these challenges requires a balanced trust framework that enhances AI literacy, promotes human oversight, and ensures the systematic adoption of responsible AI principles.

---

## Key Design Decisions

**Why RAG and not fine-tuning?**
The corpus is small (20 papers), changes over time, and answers need to be traceable to specific sources. RAG is the right tool — fine-tuning would bake knowledge in without citation ability and would need retraining whenever new papers are added.

**Why two-stage retrieval?**
Running a cross-encoder on all chunks per query would be too slow. Bi-encoder embeddings do a fast broad search first, then the cross-encoder does accurate pair-level scoring on only the top 20 candidates.

**Why diversity enforcement?**
Without it, one highly relevant paper dominates all top-K results. Capping at 2 chunks per paper forces the system to surface perspectives from multiple sources — more useful and more honest.

**Why pymupdf4llm over raw PyMuPDF?**
Raw PDF extraction produces noisy text full of headers, footers, and layout artifacts. pymupdf4llm outputs clean markdown that handles multi-column layouts, figure captions, and running headers automatically — significantly improving chunk quality.

**Hallucination prevention via score gating**
Before calling the LLM, the system checks the top rerank score. If it falls below a threshold, the question is likely outside the corpus scope and the system returns a clear "out of scope" message instead of calling the LLM at all. This prevents the model from answering confidently with fabricated citations when it has no grounding.

**Scalable ingestion**
The ingestion script checks existing chunk IDs in ChromaDB before embedding. Adding new papers means dropping PDFs in the folder and rerunning — only the new chunks get embedded.

---

## Known Limitations

- **Fixed-size chunking** can cut mid-sentence on long complex sentences. `SemanticChunker` (LangChain) would be a proper fix — splits on topic boundaries rather than character count.
- **PDF extraction quality** varies across publishers — some papers require title override logic due to nonstandard layouts.

---

## Roadmap

- [x] Data collection — arXiv API + manual curation
- [x] PDF extraction and cleaning with pymupdf4llm
- [x] Chunking with overlap and chunk quality filtering
- [x] Embedding with sentence-transformers
- [x] Persistent vector store (ChromaDB)
- [x] Two-stage retrieval with cross-encoder reranking
- [x] Diversity enforcement (max 2 chunks per paper)
- [x] Grounded generation with citations (Groq / Llama 3.1)
- [x] Hallucination prevention via rerank score gating
- [x] Stress testing — hallucination, out-of-scope, conflicting info, multi-hop
- [x] Streamlit UI with streaming responses
- [x] Docker containerization
- [x] Deploy on Hugging Face Spaces
- [x] GitHub Actions CI/CD


---

## Setup

**Clone and install:**
```bash
git clone https://github.com/Donna737/responsible-ai-rag.git
cd responsible-ai-rag
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt
```

**Set your API key:**
```bash
cp env.example .env
# add your Groq API key to .env
```
Get a free key at [console.groq.com](https://console.groq.com)

**Add your papers:**
Place PDF files in the `papers/` folder, then run:
```bash
streamlit run app.py
```
On first run the vector store builds automatically. Every run after loads instantly.

---
