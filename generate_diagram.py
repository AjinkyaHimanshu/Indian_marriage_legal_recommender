"""
Generates architecture.png — full system architecture diagram for the
Indian Marriage Law RAG project.

Run: python generate_diagram.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

# ── Colour palette ────────────────────────────────────────────────────────────
C = {
    "data_bg":        "#DBEAFE",  # blue-100
    "data_border":    "#1D4ED8",  # blue-700

    "db_bg":          "#D1FAE5",  # green-100
    "db_border":      "#065F46",  # green-800

    "retrieval_bg":   "#FEF3C7",  # amber-100
    "retrieval_border":"#92400E", # amber-800

    "gen_bg":         "#EDE9FE",  # violet-100
    "gen_border":     "#4C1D95",  # violet-900

    "eval_bg":        "#FCE7F3",  # pink-100
    "eval_border":    "#831843",  # pink-900

    "app_bg":         "#CCFBF1",  # teal-100
    "app_border":     "#134E4A",  # teal-900

    "model_bg":       "#FFF7ED",  # orange-50
    "model_border":   "#C2410C",  # orange-700

    "arrow":          "#374151",  # gray-700
    "section_label":  "#111827",  # gray-900
    "box_text":       "#1F2937",  # gray-800
    "white":          "#FFFFFF",
    "light_gray":     "#F9FAFB",
    "divider":        "#D1D5DB",
}

FIG_W, FIG_H = 22, 28


def box(ax, x, y, w, h, label, sublabel=None,
        bg="#FFFFFF", border="#000000", fontsize=9, bold=False,
        sublabel_size=7.5, radius=0.25):
    """Draw a rounded rectangle with centred label (and optional sublabel)."""
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        linewidth=1.4,
        edgecolor=border,
        facecolor=bg,
        zorder=2,
    )
    ax.add_patch(patch)
    cy = y + h / 2 + (0.12 if sublabel else 0)
    ax.text(
        x + w / 2, cy, label,
        ha="center", va="center",
        fontsize=fontsize,
        fontweight="bold" if bold else "normal",
        color=C["box_text"],
        zorder=3,
        wrap=False,
    )
    if sublabel:
        ax.text(
            x + w / 2, y + h / 2 - 0.22, sublabel,
            ha="center", va="center",
            fontsize=sublabel_size,
            color="#6B7280",
            zorder=3,
            style="italic",
        )


def arrow(ax, x0, y0, x1, y1, label=None, color=None):
    """Draw an annotated arrow between two points."""
    c = color or C["arrow"]
    ax.annotate(
        "",
        xy=(x1, y1), xytext=(x0, y0),
        arrowprops=dict(
            arrowstyle="-|>",
            color=c,
            lw=1.6,
            mutation_scale=14,
        ),
        zorder=4,
    )
    if label:
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        ax.text(mx + 0.08, my, label,
                fontsize=7, color="#6B7280", va="center", zorder=5)


def section_header(ax, x, y, w, h, title, bg, border):
    """Filled section-header bar."""
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0,rounding_size=0.15",
        linewidth=1.6,
        edgecolor=border,
        facecolor=bg,
        zorder=2,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, title,
            ha="center", va="center",
            fontsize=10.5, fontweight="bold",
            color=C["white"], zorder=3)


def divider(ax, y, x0=0.4, x1=21.6):
    ax.plot([x0, x1], [y, y], color=C["divider"], lw=0.8, ls="--", zorder=1)


# ═══════════════════════════════════════════════════════════════════════════════
# Build figure
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
ax.set_xlim(0, FIG_W)
ax.set_ylim(0, FIG_H)
ax.axis("off")
ax.set_facecolor(C["light_gray"])
fig.patch.set_facecolor(C["light_gray"])

# ── Master title ──────────────────────────────────────────────────────────────
ax.text(FIG_W / 2, 27.4,
        "Indian Marriage Law — RAG System Architecture",
        ha="center", va="center",
        fontsize=17, fontweight="bold", color=C["section_label"])
ax.text(FIG_W / 2, 27.0,
        "AAI-590 Group 16  |  University of San Diego",
        ha="center", va="center",
        fontsize=10, color="#6B7280")

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1 — DATA COLLECTION  (y 24.5 → 26.3)
# ─────────────────────────────────────────────────────────────────────────────
section_header(ax, 0.4, 25.9, 21.2, 0.65,
               "PHASE 1 — Data Collection  (Notebook 1)",
               bg=C["data_border"], border=C["data_border"])

box(ax, 0.7,  24.5, 3.6, 1.1, "India Code\n(indiacode.nic.in)",
    sublabel="Law name discovery",
    bg=C["data_bg"], border=C["data_border"])

box(ax, 5.5,  24.5, 3.6, 1.1, "Indian Kanoon\n(indiankanoon.org)",
    sublabel="Full-text HTML",
    bg=C["data_bg"], border=C["data_border"])

box(ax, 10.3, 24.5, 3.6, 1.1, "Fuzzy Name\nMatching",
    sublabel="rapidfuzz token_set_ratio",
    bg=C["data_bg"], border=C["data_border"])

box(ax, 15.1, 24.5, 3.6, 1.1, "40+ Law JSON Files",
    sublabel="data/base_chunked_marriage_laws/",
    bg=C["data_bg"], border=C["data_border"], bold=True)

arrow(ax, 4.3,  25.05, 5.5,  25.05)
arrow(ax, 9.1,  25.05, 10.3, 25.05)
arrow(ax, 13.9, 25.05, 15.1, 25.05)

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2 — VECTOR DATABASE  (y 21.0 → 24.0)
# ─────────────────────────────────────────────────────────────────────────────
section_header(ax, 0.4, 23.6, 21.2, 0.65,
               "PHASE 2 — Vector Database  (Notebooks 2 & 3)",
               bg=C["db_border"], border=C["db_border"])

# corpus box
box(ax, 0.7, 21.2, 3.2, 2.1, "1,174 Legal\nSections",
    sublabel="chunked_laws.json",
    bg=C["db_bg"], border=C["db_border"])

# unchunked path
box(ax, 5.2, 22.3, 3.4, 0.85, "jina-embeddings-v3",
    sublabel="1024-dim · 8192 tokens",
    bg=C["model_bg"], border=C["model_border"])
box(ax, 5.2, 21.2, 3.4, 0.85, "all-MiniLM-L6-v2",
    sublabel="384-dim · 512 tokens",
    bg=C["model_bg"], border=C["model_border"])

# sliding window
box(ax, 10.0, 22.3, 3.4, 0.85, "Sliding Window\nChunker",
    sublabel="400 tok · 50 overlap",
    bg=C["db_bg"], border=C["db_border"])
box(ax, 10.0, 21.2, 3.4, 0.85, "Sliding Window\nChunker",
    sublabel="1024 tok · 50 overlap",
    bg=C["db_bg"], border=C["db_border"])

# qdrant collections
box(ax, 14.8, 22.85, 3.8, 0.9, "legal_acts_baseline",
    sublabel="384-dim · unchunked",
    bg=C["db_bg"], border=C["db_border"])
box(ax, 14.8, 23.9, 3.8, 0.9, "legal_acts_jina",        # overlap with header
    sublabel="1024-dim · unchunked",
    bg=C["db_bg"], border=C["db_border"])

# Actually let me redo the layout for phase 2 more carefully

# Let me just put 4 Qdrant collections on the right side
box(ax, 14.8, 22.85, 6.0, 0.7, "legal_acts_jina  (1024-dim, unchunked)",
    sublabel="Notebook 2",
    bg=C["db_bg"], border=C["db_border"], fontsize=8.5)
box(ax, 14.8, 21.9,  6.0, 0.7, "legal_acts_baseline  (384-dim, unchunked)",
    sublabel="Notebook 2",
    bg=C["db_bg"], border=C["db_border"], fontsize=8.5)
box(ax, 14.8, 21.0,  6.0, 0.7, "legal_jina_chunked  (1024-dim, chunked)",
    sublabel="Notebook 3",
    bg=C["db_bg"], border=C["db_border"], fontsize=8.5)

# arrows data → corpus
arrow(ax, 16.9, 24.5, 2.0, 23.3,  color=C["db_border"])

# corpus → encoders
arrow(ax, 3.9, 22.75, 5.2, 22.75)
arrow(ax, 3.9, 21.65, 5.2, 21.65)

# Jina encoder → unchunked collection
arrow(ax, 8.6, 22.75, 14.8, 23.25)
# Baseline → unchunked
arrow(ax, 8.6, 21.65, 14.8, 22.25)

# corpus → chunker
arrow(ax, 3.9, 22.0, 10.0, 22.75)
arrow(ax, 3.9, 22.0, 10.0, 21.65)

# chunker → chunked collections
arrow(ax, 13.4, 22.75, 14.8, 21.35)
arrow(ax, 13.4, 21.65, 14.8, 21.35)

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 3 — QUERY EXPANSION  (y 19.2 → 20.8)
# ─────────────────────────────────────────────────────────────────────────────
section_header(ax, 0.4, 20.5, 21.2, 0.65,
               "PHASE 3 — Query Expansion  (Notebook 4)",
               bg="#7C3AED", border="#7C3AED")

box(ax, 0.7,  19.2, 3.6, 1.0, "benchmarking_data.csv",
    sublabel="15 benchmark questions",
    bg=C["gen_bg"], border=C["gen_border"])
box(ax, 6.0,  19.2, 4.0, 1.0, "Claude (claude-sonnet-4-6)",
    sublabel="Legal terminology expansion",
    bg=C["model_bg"], border=C["model_border"])
box(ax, 12.0, 19.2, 4.2, 1.0, "benchmarking_data_expanded.csv",
    sublabel="+ Expanded_Query column",
    bg=C["gen_bg"], border=C["gen_border"], bold=True)

arrow(ax, 4.3,  19.7, 6.0,  19.7)
arrow(ax, 10.0, 19.7, 12.0, 19.7)

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 4 — RETRIEVAL  (y 15.2 → 19.0)  Machine 2
# ─────────────────────────────────────────────────────────────────────────────
section_header(ax, 0.4, 18.55, 21.2, 0.65,
               "PHASE 4 — Retrieval  (Notebook 5 · Machine 2 — needs models + Qdrant)",
               bg=C["retrieval_border"], border=C["retrieval_border"])

# --- Unchunked path (top row) ---
box(ax, 0.7,  17.3, 3.0, 0.9, "Original Question",
    sublabel="Question column",
    bg=C["retrieval_bg"], border=C["retrieval_border"], fontsize=8.5)
box(ax, 4.5,  17.3, 3.2, 0.9, "Vector Search",
    sublabel="Cosine similarity · top-5",
    bg=C["retrieval_bg"], border=C["retrieval_border"], fontsize=8.5)
box(ax, 8.5,  17.3, 3.4, 0.9, "Jina Unchunked\nContexts",
    sublabel="jina_unchunked_contexts.json",
    bg=C["retrieval_bg"], border=C["retrieval_border"], fontsize=8.5)
box(ax, 13.2, 17.3, 3.4, 0.9, "Baseline Unchunked\nContexts",
    sublabel="baseline_unchunked_contexts.json",
    bg=C["retrieval_bg"], border=C["retrieval_border"], fontsize=8.5)

arrow(ax, 3.7,  17.75, 4.5,  17.75)
arrow(ax, 7.7,  17.75, 8.5,  17.75)
arrow(ax, 7.7,  17.75, 13.2, 17.75)

# --- Rechunked path (bottom row) ---
box(ax, 0.7,  15.8, 3.0, 0.9, "Expanded Query",
    sublabel="Expanded_Query column",
    bg=C["retrieval_bg"], border=C["retrieval_border"], fontsize=8.5)

# hybrid pipeline sub-boxes
box(ax, 4.5,  15.5, 2.2, 0.7, "Vector Search",
    bg=C["retrieval_bg"], border=C["retrieval_border"], fontsize=8)
box(ax, 4.5,  16.3, 2.2, 0.7, "BM25",
    sublabel="Keyword matching",
    bg=C["retrieval_bg"], border=C["retrieval_border"], fontsize=8)
box(ax, 7.2,  15.8, 2.8, 0.9, "Cross-Encoder\nRe-ranking",
    sublabel="0.3·BM25 + 0.7·CE",
    bg=C["model_bg"], border=C["model_border"], fontsize=8.5)
box(ax, 10.5, 15.8, 2.5, 0.9, "Parent-Child\nMerge",
    sublabel="Reconstruct sections",
    bg=C["retrieval_bg"], border=C["retrieval_border"], fontsize=8.5)

box(ax, 13.8, 15.5, 3.0, 0.7, "Jina Rechunked\nContexts",
    sublabel="jina_rechunked_contexts.json",
    bg=C["retrieval_bg"], border=C["retrieval_border"], fontsize=7.5)
box(ax, 17.3, 15.5, 3.0, 0.7, "Baseline Rechunked\nContexts",
    sublabel="baseline_rechunked_contexts.json",
    bg=C["retrieval_bg"], border=C["retrieval_border"], fontsize=7.5)

arrow(ax, 3.7,  16.25, 4.5,  16.65)
arrow(ax, 3.7,  16.25, 4.5,  15.85)
arrow(ax, 6.7,  16.25, 7.2,  16.25)
arrow(ax, 6.7,  15.85, 7.2,  16.25)
arrow(ax, 10.0, 16.25, 10.5, 16.25)
arrow(ax, 13.0, 16.25, 13.8, 15.85)
arrow(ax, 13.0, 16.25, 17.3, 15.85)

# Qdrant DB feeding retrieval
ax.text(17.5, 18.2, "Qdrant\nVector DB", ha="center", va="center",
        fontsize=8.5, fontweight="bold", color=C["db_border"],
        bbox=dict(fc=C["db_bg"], ec=C["db_border"], boxstyle="round,pad=0.3", lw=1.2))

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 5 — GENERATION & EVALUATION  (y 11.5 → 15.0)  Machine 1
# ─────────────────────────────────────────────────────────────────────────────
section_header(ax, 0.4, 14.8, 21.2, 0.65,
               "PHASE 5 — Generation & Evaluation  (Notebook 6 · Machine 1 — Anthropic API only)",
               bg=C["gen_border"], border=C["gen_border"])

# 4 context inputs
box(ax, 0.5,  13.5, 2.4, 0.9, "Jina\nUnchunked",     bg=C["retrieval_bg"], border=C["retrieval_border"], fontsize=8)
box(ax, 3.2,  13.5, 2.4, 0.9, "Baseline\nUnchunked",  bg=C["retrieval_bg"], border=C["retrieval_border"], fontsize=8)
box(ax, 5.9,  13.5, 2.4, 0.9, "Jina\nRechunked",      bg=C["retrieval_bg"], border=C["retrieval_border"], fontsize=8)
box(ax, 8.6,  13.5, 2.4, 0.9, "Baseline\nRechunked",  bg=C["retrieval_bg"], border=C["retrieval_border"], fontsize=8)

# Generation + evaluation nodes
box(ax, 0.5,  12.1, 4.5, 1.0, "Answer Generation",
    sublabel="Claude (temperature 0.2)",
    bg=C["gen_bg"], border=C["gen_border"])
box(ax, 5.7,  12.1, 4.5, 1.0, "Relevance Scoring",
    sublabel="LLM-as-a-Judge  (0–5 per chunk)",
    bg=C["gen_bg"], border=C["gen_border"])
box(ax, 11.1, 12.1, 3.5, 1.0, "nDCG Calculation",
    sublabel="Per query · normalised DCG",
    bg=C["gen_bg"], border=C["gen_border"])

# outputs
box(ax, 0.5,  11.0, 4.0, 0.8, "*_final.json",
    sublabel="results/evaluated/",
    bg=C["gen_bg"], border=C["gen_border"], fontsize=8.5)
box(ax, 5.5,  11.0, 4.5, 0.8, "evaluation_results.xlsx",
    sublabel="1 row per chunk · merged cells",
    bg=C["gen_bg"], border=C["gen_border"], fontsize=8.5)
box(ax, 11.1, 11.0, 3.5, 0.8, "Checkpoint Cache",
    sublabel="Skip already-evaluated rows",
    bg=C["gen_bg"], border=C["gen_border"], fontsize=8.5)

# arrows
for cx in [1.7, 4.4, 7.1, 9.8]:
    arrow(ax, cx, 13.5, cx, 13.1)

arrow(ax, 2.75, 12.1, 2.75, 11.8)
arrow(ax, 7.95, 12.1, 7.95, 11.8)
arrow(ax, 12.85, 12.1, 12.85, 11.8)
arrow(ax, 5.0,  12.6, 5.7,  12.6)
arrow(ax, 10.2, 12.6, 11.1, 12.6)

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 6 — EVALUATION PLOTS  (y 8.5 → 10.6)
# ─────────────────────────────────────────────────────────────────────────────
section_header(ax, 0.4, 10.3, 21.2, 0.65,
               "PHASE 6 — Evaluation & Comparison  (Notebook 7)",
               bg=C["eval_border"], border=C["eval_border"])

plots = [
    ("Avg nDCG\nBar Chart",     "plot_avg_ndcg.png"),
    ("nDCG per Query\nLine Chart", "plot_perquery_ndcg.png"),
    ("nDCG Distribution\nBox Plot",  "plot_ndcg_boxplot.png"),
    ("Avg Relevance\nBar Chart",  "plot_avg_relevance.png"),
    ("nDCG vs Relevance\nScatter",  "plot_ndcg_vs_relevance.png"),
]
for i, (label, fname) in enumerate(plots):
    bx = 0.7 + i * 4.2
    box(ax, bx, 8.7, 3.8, 1.3, label,
        sublabel=fname,
        bg=C["eval_bg"], border=C["eval_border"], fontsize=8.5)

# summary table
box(ax, 0.7, 8.0, 19.5, 0.55, "Summary Table — Avg nDCG · Std nDCG · Min/Max nDCG · Avg Relevance · Avg Latency · Config Recommendation",
    bg=C["eval_bg"], border=C["eval_border"], fontsize=8.5)

# arrow from evaluated JSON to plots
arrow(ax, 2.5, 11.0, 5.0, 10.0, color=C["eval_border"])

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 7 — STREAMLIT APP  (y 4.5 → 7.5)
# ─────────────────────────────────────────────────────────────────────────────
section_header(ax, 0.4, 7.55, 21.2, 0.65,
               "STREAMLIT APP  (app.py · Jina Rechunked config)",
               bg=C["app_border"], border=C["app_border"])

boxes_app = [
    ("User Question",         "Text area input",             0.6),
    ("Query Expansion",       "Claude (temp 0.2)",           3.8),
    ("Vector Search",         "jina-embeddings-v3",          7.0),
    ("BM25 + CrossEncoder",   "Hybrid re-ranking",           10.2),
    ("Parent-Child Merge",    "Reconstruct sections",        13.4),
    ("Answer Generation",     "Claude (temp 0.2)",           16.6),
    ("Legal Answer",          "Grounded response",           19.8),
]
for label, sub, bx in boxes_app:
    box(ax, bx, 5.8, 2.9, 1.4, label,
        sublabel=sub,
        bg=C["app_bg"], border=C["app_border"], fontsize=8.5)

for i in range(len(boxes_app) - 1):
    ax.annotate("",
        xy=(boxes_app[i+1][2] + 0.05, 6.5),
        xytext=(boxes_app[i][2] + 2.9 - 0.05, 6.5),
        arrowprops=dict(arrowstyle="-|>", color=C["app_border"],
                        lw=1.6, mutation_scale=12),
        zorder=4)

# LLM provider badge
box(ax, 0.7, 4.5, 9.0, 0.95,
    "LLM Provider: Anthropic Claude (claude-sonnet-4-6)  ·  Framework: LangChain",
    sublabel="Configured via ANTHROPIC_API_KEY or config.yaml",
    bg=C["model_bg"], border=C["model_border"], fontsize=9)

box(ax, 10.5, 4.5, 9.0, 0.95,
    "Qdrant local DB  ·  legal_jina_chunked collection",
    sublabel="database/qdrant_db/  ·  1024-dim vectors  ·  sliding-window chunks",
    bg=C["db_bg"], border=C["db_border"], fontsize=9)

# ─────────────────────────────────────────────────────────────────────────────
# Legend
# ─────────────────────────────────────────────────────────────────────────────
legend_items = [
    (C["data_bg"],       C["data_border"],       "Data Collection"),
    (C["db_bg"],         C["db_border"],          "Vector Database"),
    (C["retrieval_bg"],  C["retrieval_border"],   "Retrieval"),
    (C["gen_bg"],        C["gen_border"],          "Generation / Evaluation"),
    (C["eval_bg"],       C["eval_border"],         "Evaluation Plots"),
    (C["app_bg"],        C["app_border"],          "Streamlit App"),
    (C["model_bg"],      C["model_border"],        "Model / LLM"),
]

lx, ly = 0.7, 3.4
ax.text(lx, ly + 0.3, "Legend", fontsize=9, fontweight="bold", color=C["section_label"])
for i, (bg, border, label) in enumerate(legend_items):
    ix = lx + i * 3.0
    patch = FancyBboxPatch((ix, ly - 0.35), 0.45, 0.38,
                           boxstyle="round,pad=0,rounding_size=0.05",
                           fc=bg, ec=border, lw=1.2, zorder=2)
    ax.add_patch(patch)
    ax.text(ix + 0.55, ly - 0.16, label, fontsize=7.5,
            va="center", color=C["box_text"])

# ─────────────────────────────────────────────────────────────────────────────
# Machine boundary annotation
# ─────────────────────────────────────────────────────────────────────────────
for y_pos, label, color in [
    (18.5, "◀  Machine 2  (GPU/RAM for models + local Qdrant)", C["retrieval_border"]),
    (14.75, "◀  Machine 1  (Anthropic API only, no local models needed)", C["gen_border"]),
]:
    ax.text(21.55, y_pos, label,
            fontsize=7.5, color=color, ha="right", va="center",
            style="italic",
            bbox=dict(fc="white", ec=color, boxstyle="round,pad=0.2", lw=0.8, alpha=0.85))

# ─────────────────────────────────────────────────────────────────────────────
# Save
# ─────────────────────────────────────────────────────────────────────────────
out = "architecture.png"
plt.tight_layout(pad=0.3)
plt.savefig(out, dpi=160, bbox_inches="tight",
            facecolor=C["light_gray"], edgecolor="none")
plt.close()
print(f"Saved -> {out}")
