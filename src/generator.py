"""
generator.py
------------
Sends retrieved chunks as context to the LLM and generates
a grounded answer with inline citations.

Supports both regular and streaming generation.
"""

from groq import Groq
from src.config import GROQ_API_KEY, GROQ_MODEL


# ── context builder ──────────────────────────────────────────
def build_context(chunks):
    """Format reranked chunks into a numbered context string."""
    parts = []
    for i, chunk in enumerate(chunks, 1):
        parts.append(f"[Source {i}: {chunk['title']}]\n{chunk['text']}")
    return "\n\n---\n\n".join(parts)


# ── prompts ──────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert research assistant specializing in Responsible AI.
You have been given context from a curated set of academic research papers.
Your goal is to provide thorough, insightful, and well-cited answers.

ANSWER QUALITY:
- Give detailed, analytical answers that synthesize insights across sources
- Connect ideas from different papers where relevant
- Use academic tone — precise, nuanced, and substantive
- Try to incorporate insights from as many provided sources as possible
- Longer thoughtful answers are better than short superficial ones

CITATION RULES:
- Always cite using [Source N] notation inline
- Citation format is strictly [Source N] — never [Source N: anything]
- Never append author names, years, or extra text inside citation brackets
- Never append subsection numbers, page numbers, or section references inside citation brackets
- [Source N] is the complete citation format — nothing else goes inside the brackets

STRICT LIMITS:
- Use ONLY the provided context — never external knowledge
- Never fabricate or hallucinate citations
- Only say the corpus is insufficient if NO sources contain ANY relevant information
- If sources contain partial information, use it and build the best answer you can"""

def build_user_prompt(question, context):
    return f"""Context from research papers:

{context}

Question: {question}

Answer with inline citations:"""


# ── standard generation ──────────────────────────────────────
def generate(question, chunks):
    """
    Generate a grounded answer from reranked chunks.
    Returns the full answer string.
    """
    context  = build_context(chunks)
    client   = Groq(api_key=GROQ_API_KEY)
    response = client.chat.completions.create(
        model    = GROQ_MODEL,
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": build_user_prompt(question, context)},
        ],
        temperature = 0.1,
        max_tokens  = 1024,
    )
    return response.choices[0].message.content


# ── streaming generation ─────────────────────────────────────
def generate_streaming(question, chunks):
    """
    Generator function for streaming responses.
    Yields tokens one by one as they arrive from the API.
    Use in Streamlit with: for token in generate_streaming(...): ...
    """
    context = build_context(chunks)
    client  = Groq(api_key=GROQ_API_KEY)
    stream  = client.chat.completions.create(
        model    = GROQ_MODEL,
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": build_user_prompt(question, context)},
        ],
        temperature = 0.1,
        max_tokens  = 1024,
        stream      = True,
    )
    for chunk in stream:
        token = chunk.choices[0].delta.content
        if token:
            yield token
