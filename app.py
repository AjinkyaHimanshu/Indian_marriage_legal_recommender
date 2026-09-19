"""Indian Marriage Law Assistant — Streamlit UI (thin orchestrator).

All logic lives in the ``indian_marriage_legal_recommender`` package; this file only
wires the LangChain + Claude pipeline to a UI.

    Configuration : Jina Rechunked (jina-embeddings-v3 + hybrid retrieval)
    Framework     : LangChain
    LLM           : Anthropic Claude (claude-sonnet-4-6)

Run with:  streamlit run app.py
"""

# ── SSL fix for corporate proxy / self-signed certificate environments ────────
import ssl
import warnings

import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings("ignore", message="Unverified HTTPS request")
ssl._create_default_https_context = ssl._create_unverified_context

_orig_create_ctx = ssl.create_default_context


def _no_verify_ctx(*args, **kwargs):
    ctx = _orig_create_ctx(*args, **kwargs)
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


ssl.create_default_context = _no_verify_ctx

from requests.adapters import HTTPAdapter  # noqa: E402

_orig_send = HTTPAdapter.send


def _patched_send(self, request, **kwargs):
    kwargs["verify"] = False
    return _orig_send(self, request, **kwargs)


HTTPAdapter.send = _patched_send
# ──────────────────────────────────────────────────────────────────────────────

import os  # noqa: E402

import streamlit as st  # noqa: E402

from config import config  # noqa: E402
from indian_marriage_legal_recommender.pipeline import answer  # noqa: E402

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

st.set_page_config(page_title="Indian Marriage Law Assistant", page_icon="⚖️", layout="wide")

# ── Verify the Claude key up front ────────────────────────────────────────────
try:
    config.require_anthropic_key()
except RuntimeError as exc:
    st.error(str(exc))
    st.stop()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚖️ Indian Marriage Law Assistant")
    st.markdown("---")
    st.markdown("### Configuration")
    st.markdown(
        f"""
| Setting | Value |
|---------|-------|
| **Config** | Jina Rechunked |
| **Framework** | `LangChain` |
| **Embedding** | `jina-embeddings-v3` |
| **Collection** | `{config.profile('jina_rechunked').collection}` |
| **Retrieval** | Vector + BM25 + CrossEncoder |
| **LLM Provider** | `anthropic` |
| **LLM Model** | `{config.CLAUDE_MODEL}` |
"""
    )
    st.markdown("---")
    st.markdown("### How it works")
    st.markdown(
        "Your question is **expanded** into legal terminology, then a **hybrid search** "
        "(vector + BM25 + Cross-Encoder) retrieves the most relevant legal sections. "
        "Those sections are reconstructed from their chunk fragments and passed to Claude, "
        "which answers **grounded strictly in the retrieved law** — no hallucination."
    )
    st.markdown("---")
    top_k = st.slider("Retrieved sections (top-k)", 1, 10, config.TOP_K_FINAL)
    show_contexts = st.toggle("Show retrieved legal sections", value=True)
    show_expanded = st.toggle("Show expanded query", value=False)

# ── Main area ───────────────────────────────────────────────────────────────
st.header("Ask a question about Indian Marriage Law")
st.caption(
    "Covers: Hindu Marriage Act, Special Marriage Act, Muslim Personal Law, Parsi "
    "Marriage Act, Christian Marriage Act, Dissolution of Muslim Marriages Act, "
    "Prohibition of Child Marriage Act, Foreign Marriage Act, and more."
)

question = st.text_area(
    "Your question",
    placeholder="e.g. I am a Hindu woman from Kerala and my partner is a Christian man. "
    "Can we marry without converting?",
    height=100,
)

if st.button("Ask ⚖️", type="primary"):
    if not question.strip():
        st.warning("Please enter a question before submitting.")
        st.stop()

    with st.status("Processing your question…", expanded=True) as status:
        st.write("🔍 Expanding query · 📚 Retrieving sections · ✍️ Generating answer…")
        result = answer(question, profile_key="jina_rechunked", top_k=top_k)
        t = result.timings
        status.update(
            label=(
                f"Done — expanded in {t['expand']:.1f}s · "
                f"retrieved in {t['retrieve']:.1f}s · generated in {t['generate']:.1f}s"
            ),
            state="complete",
        )

    st.markdown("---")
    st.subheader("📋 Legal Answer")
    st.markdown(result.answer)

    if show_expanded and result.expanded_query != question:
        with st.expander("🔎 Expanded query used for retrieval"):
            st.info(result.expanded_query)

    if show_contexts and result.documents:
        st.markdown("---")
        st.subheader(f"📖 Retrieved Legal Sections ({len(result.documents)} found)")
        for i, doc in enumerate(result.documents, 1):
            title = doc.metadata.get("section", f"Section {i}")
            with st.expander(f"Section {i} — {title}", expanded=(i == 1)):
                st.code(doc.page_content, language=None)
    elif not result.documents:
        st.warning(
            "No relevant legal sections were retrieved. "
            "The topic may not be covered in the current corpus."
        )
