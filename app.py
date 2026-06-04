"""
app.py
------
Streamlit UI for the Responsible AI Research Assistant.

Run with:
    streamlit run app.py
"""

import streamlit as st
from src.ingest import run_ingestion
from src.retriever import load_retriever, retrieve, rerank
from src.generator import generate_streaming

# ── page config ──────────────────────────────────────────────
st.set_page_config(
    page_title = "Responsible AI Research Assistant",
    page_icon  = "◈",
    layout     = "wide",
)

# ── custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@300;400;500&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0d0f12;
    color: #e8e4dc;
}
.stApp { background: #0d0f12; }
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }
.main .block-container { max-width: 860px; padding: 3rem 2rem; margin: 0 auto; }

.rai-header { border-bottom: 1px solid #2a2d35; padding-bottom: 2rem; margin-bottom: 2.5rem; }
.rai-title { font-family: 'DM Serif Display', serif; font-size: 2.4rem; font-weight: 400; color: #e8e4dc; letter-spacing: -0.02em; line-height: 1.1; margin: 0; }
.rai-title em { font-style: italic; color: #c8b89a; }
.rai-subtitle { font-family: 'DM Mono', monospace; font-size: 0.72rem; color: #5a5f6e; letter-spacing: 0.12em; text-transform: uppercase; margin-top: 0.6rem; }

.stats-bar { display: flex; gap: 2rem; margin-bottom: 2rem; padding: 1rem 1.2rem; background: #13151a; border: 1px solid #1e2128; border-radius: 4px; }
.stat-item { display: flex; flex-direction: column; gap: 2px; }
.stat-value { font-family: 'DM Serif Display', serif; font-size: 1.3rem; color: #c8b89a; }
.stat-label { font-family: 'DM Mono', monospace; font-size: 0.65rem; color: #5a5f6e; text-transform: uppercase; letter-spacing: 0.1em; }

.stTextArea textarea { background: #13151a !important; border: 1px solid #2a2d35 !important; border-radius: 4px !important; color: #e8e4dc !important; font-family: 'DM Sans', sans-serif !important; font-size: 0.95rem !important; padding: 1rem !important; resize: none !important; }
.stTextArea textarea:focus { border-color: #c8b89a !important; box-shadow: none !important; }
.stTextArea textarea::placeholder { color: #3a3f4a !important; }

.stButton button { background: #c8b89a !important; color: #0d0f12 !important; border: none !important; border-radius: 3px !important; font-family: 'DM Mono', monospace !important; font-size: 0.75rem !important; font-weight: 500 !important; letter-spacing: 0.1em !important; text-transform: uppercase !important; padding: 0.6rem 1.8rem !important; }
.stButton button:hover { opacity: 0.85 !important; }

.answer-container { background: #13151a; border: 1px solid #1e2128; border-left: 3px solid #c8b89a; border-radius: 4px; padding: 1.5rem 1.8rem; margin: 1.5rem 0; font-size: 0.95rem; line-height: 1.75; color: #d4cfc6; }
.sources-header { font-family: 'DM Mono', monospace; font-size: 0.68rem; color: #5a5f6e; letter-spacing: 0.12em; text-transform: uppercase; margin: 1.5rem 0 0.8rem; }
.source-card { background: #13151a; border: 1px solid #1e2128; border-radius: 4px; padding: 0.8rem 1rem; margin-bottom: 0.5rem; display: flex; align-items: flex-start; gap: 0.8rem; }
.source-number { font-family: 'DM Serif Display', serif; font-size: 1.1rem; color: #c8b89a; min-width: 1.5rem; line-height: 1.3; }
.source-info { flex: 1; }
.source-title { font-size: 0.85rem; color: #e8e4dc; font-weight: 500; line-height: 1.3; }
.source-meta { font-family: 'DM Mono', monospace; font-size: 0.65rem; color: #5a5f6e; margin-top: 3px; }
.score-pill { font-family: 'DM Mono', monospace; font-size: 0.62rem; color: #5a5f6e; background: #0d0f12; border: 1px solid #1e2128; border-radius: 2px; padding: 2px 6px; white-space: nowrap; }
.examples-header { font-family: 'DM Mono', monospace; font-size: 0.68rem; color: #5a5f6e; letter-spacing: 0.12em; text-transform: uppercase; margin-bottom: 0.8rem; }
hr { border: none; border-top: 1px solid #1e2128; margin: 2rem 0; }
.stSpinner > div { border-top-color: #c8b89a !important; }
</style>
""", unsafe_allow_html=True)


# ── load models (cached — runs once when app starts) ─────────
@st.cache_resource
def get_retriever():
    """
    Load models and connect to ChromaDB.
    st.cache_resource ensures this runs only once per app session
    even if the user asks multiple questions.
    If vector store doesn't exist, runs ingestion first.
    """
    from pathlib import Path
    from src.config import CHROMA_DIR, COLLECTION_NAME
    import chromadb

    chroma_path = Path(CHROMA_DIR)
    needs_ingestion = False

    if not chroma_path.exists():
        needs_ingestion = True
    else:
        try:
            client     = chromadb.PersistentClient(path=str(CHROMA_DIR))
            collection = client.get_collection(COLLECTION_NAME)
            if collection.count() == 0:
                needs_ingestion = True
        except Exception:
            needs_ingestion = True

    if needs_ingestion:
        with st.spinner("First run — building vector store from papers. This takes ~10 minutes..."):
            run_ingestion()

    return load_retriever()


embedder, reranker, collection = get_retriever()


# ── header ───────────────────────────────────────────────────
st.markdown("""
<div class="rai-header">
    <h1 class="rai-title">Responsible AI<br><em>Research Assistant</em></h1>
    <p class="rai-subtitle">Retrieval-Augmented Generation · 20 Research Papers · arXiv Corpus</p>
</div>
""", unsafe_allow_html=True)

# ── stats bar ────────────────────────────────────────────────
total_chunks = collection.count()
st.markdown(f"""
<div class="stats-bar">
    <div class="stat-item">
        <span class="stat-value">20</span>
        <span class="stat-label">Papers indexed</span>
    </div>
    <div class="stat-item">
        <span class="stat-value">{total_chunks:,}</span>
        <span class="stat-label">Chunks embedded</span>
    </div>
    <div class="stat-item">
        <span class="stat-value">2</span>
        <span class="stat-label">Stage retrieval</span>
    </div>
    <div class="stat-item">
        <span class="stat-value">2020–2024</span>
        <span class="stat-label">Coverage</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── example questions ────────────────────────────────────────
st.markdown('<p class="examples-header">Example questions</p>', unsafe_allow_html=True)

examples = [
    "What are the main challenges of responsible AI?",
    "How does bias affect hiring decisions?",
    "Is transparency always beneficial in AI?",
    "Can AI ever be truly fair?",
    "What is the relationship between explainability and accountability?",
]

if "selected_example" not in st.session_state:
    st.session_state.selected_example = ""

cols = st.columns(len(examples))
for i, (col, example) in enumerate(zip(cols, examples)):
    with col:
        label = example[:35] + "..." if len(example) > 35 else example
        if st.button(label, key=f"ex_{i}"):
            st.session_state.selected_example = example

# ── question input ───────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)

question = st.text_area(
    label            = "Your question",
    value            = st.session_state.selected_example,
    placeholder      = "Ask anything about responsible AI, fairness, transparency, accountability...",
    height           = 100,
    label_visibility = "collapsed",
)

col1, col2 = st.columns([1, 5])
with col1:
    search = st.button("Ask", use_container_width=True)

# ── query handling ───────────────────────────────────────────
if search and question.strip():
    st.session_state.selected_example = ""

    with st.spinner("Retrieving and reranking..."):
        retrieved = retrieve(question, embedder, collection)
        reranked  = rerank(question, retrieved, reranker)

    # gate — don't call LLM if retrieval quality is too low
    # prevents hallucination on out-of-scope questions
    if not reranked or reranked[0]["rerank_score"] < 0.5:
        st.markdown("""
        <div class="answer-container">
        This question appears to be outside the scope of the responsible AI research corpus.
        Try asking about fairness, transparency, accountability, explainability, or AI ethics.
        </div>
        """, unsafe_allow_html=True)
    else:
        # streaming answer
        st.markdown('<div class="answer-container">', unsafe_allow_html=True)
        answer_placeholder = st.empty()
        full_answer = ""

        for token in generate_streaming(question, reranked):
            full_answer += token
            answer_placeholder.markdown(full_answer + "▌")

        answer_placeholder.markdown(full_answer)
        st.markdown('</div>', unsafe_allow_html=True)

        # sources
        st.markdown('<p class="sources-header">Sources retrieved</p>',
                    unsafe_allow_html=True)

        for i, chunk in enumerate(reranked, 1):
            st.markdown(f"""
            <div class="source-card">
                <span class="source-number">{i}</span>
                <div class="source-info">
                    <div class="source-title">{chunk['title']}</div>
                    <div class="source-meta">
                        Similarity: {chunk['similarity']} · Chunk {chunk.get('chunk_idx', '—')}
                    </div>
                </div>
                <span class="score-pill">score {chunk['rerank_score']}</span>
            </div>
            """, unsafe_allow_html=True)

elif search and not question.strip():
    st.warning("Please enter a question.")
# ── footer ───────────────────────────────────────────────────
st.markdown("""
<hr>
<p style="font-family: 'DM Mono', monospace; font-size: 0.65rem; color: #2a2d35; text-align: center; letter-spacing: 0.08em;">
RAG PIPELINE · SENTENCE-TRANSFORMERS · CHROMADB · GROQ · BUILT FOR PORTFOLIO
</p>
""", unsafe_allow_html=True)
