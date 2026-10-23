# Dead Code Analysis

Identified functions that are defined but never called anywhere in the project.
**Do not remove without review** — some may be kept intentionally as utilities.

---

## Unused Functions (4 found)

| # | Function | File | Line |
|---|---|---|---|
| 1 | `divider()` | `generate_diagram.py` | 119 |
| 2 | `for_profile()` | `src/indian_marriage_legal_recommender/embeddings.py` | 111 |
| 3 | `as_langchain_store()` | `src/indian_marriage_legal_recommender/vectorstore.py` | 42 |
| 4 | `RAGResult.sections` (property) | `src/indian_marriage_legal_recommender/pipeline.py` | 30 |

---

## Details

### 1. `divider()` — `generate_diagram.py:119`
```python
def divider(ax, y, x0=0.4, x1=21.6):
    """Draw a dashed horizontal divider line across the figure at height y."""
    ax.plot([x0, x1], [y, y], color=C["divider"], lw=0.8, ls="--", zorder=1)
```
**Why unused:** Defined alongside `box()`, `arrow()`, and `section_header()` but never called in the diagram generation script. The other three helpers are all used; this one was likely added as a utility and then not needed.

---

### 2. `for_profile()` — `embeddings.py:111`
```python
def for_profile(profile: "config.ConfigProfile") -> Embeddings:
    """Return the appropriate cache-first Embeddings object for the given config profile."""
    return get_embeddings(profile.embed_model)
```
**Why unused:** A thin convenience wrapper around `get_embeddings(profile.embed_model)`. All call sites use `get_embeddings()` directly (e.g. `ingest.py`, `retrieval.py`). Never imported or invoked elsewhere.

---

### 3. `as_langchain_store()` — `vectorstore.py:42`
```python
def as_langchain_store(collection: str, embed_model: str, path: Optional[str] = None):
    """Wrap a collection as a LangChain QdrantVectorStore."""
    ...
```
**Why unused:** Provides a LangChain `QdrantVectorStore` interface over a Qdrant collection, but the project queries Qdrant directly via `get_client()` and `HybridLegalRetriever`. This function was likely scaffolded for an earlier LangChain-native retrieval approach that was replaced by the custom hybrid retriever.

---

### 4. `RAGResult.sections` (property) — `pipeline.py:30`
```python
@property
def sections(self) -> List[str]:
    """Return the section header string for each retrieved document."""
    return [d.metadata.get("section", "") for d in self.documents]
```
**Why unused:** The `RAGResult` dataclass exposes `.answer`, `.documents`, `.expanded_query`, and `.timings` — all of which are accessed in `app.py`. The `.sections` property is never read. The app accesses `doc.metadata.get("section", ...)` directly inside a loop instead.

---

## Notes

- All other functions are actively used — either called directly, required by the LangChain framework (`_get_relevant_documents`, `embed_documents`, `embed_query`), or invoked from Jupyter notebooks.
- Private helpers (`_tokenizer`, `_key`, `_dcg`, etc.) were all verified as called within their own module.
