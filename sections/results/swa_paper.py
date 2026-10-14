"""
========================================================================================
Module: Results - "Sliding-Window Beats Linear Attention" (Jolicoeur-Martineau et al., 2026)
Author: Gusev Roman (Results Interpreter & Challenger)
========================================================================================
A short results summary of the second paper: setup and metrics, short-context results,
long-context tasks, and speed and memory. All numbers come from
swa_paper_data.py.
========================================================================================
"""

import re

import plotly.graph_objects as go
import streamlit as st

from sections.results.common import md_table
from sections.results.paper_data import AMBER, BLUE, GREEN, SLATE
from sections.results.swa_paper_data import TABLE_1, TABLE_2, TABLE_4, TABLE_5, TABLE_6

SWA_COLOR, TEACHER_COLOR, LINEAR_COLOR = AMBER, SLATE, BLUE
CONTEXTS_S_NIAH = ["0.5K", "1K", "2K", "4K"]


def _models(table: list[dict]) -> list[str]:
    return list(dict.fromkeys(r["model"] for r in table))


def _best_linear_row(rows: list[dict], key: str):
    """The linearized (retrofitted) method with the highest score among the rows of one base model."""
    candidates = [r for r in rows if r["kind"] == "Linearized" and r[key] is not None]
    return max(candidates, key=lambda r: r[key]) if candidates else None


def _best_linear(rows: list[dict], key: str):
    row = _best_linear_row(rows, key)
    return None if row is None else row[key]


def _short_name(method: str) -> str:
    return re.sub(r"\s*\(.*\)$", "", method)


def _method_label(method: str) -> str:
    return "SWA with sinks" if method.startswith("SWA") else ("LoLCATs" if method.startswith("LoL") else ("Liger-GLA" if method.startswith("Liger") else "Full attention"))


def _grouped_average_chart(table: list[dict], swa_method: str, title: str, y_range: list[float] | None = None) -> go.Figure:
    """Teacher vs SWA vs best linearized average score, one group of bars per base model."""
    models = _models(table)
    teacher = [next(r for r in table if r["model"] == m and r["kind"] == "Teacher")["avg"] for m in models]
    swa = [next(r for r in table if r["model"] == m and r["method"].replace(" ", "") == swa_method)["avg"] for m in models]
    best = [_best_linear_row([r for r in table if r["model"] == m], "avg") for m in models]
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Teacher (full attention)", x=models, y=teacher, marker_color=TEACHER_COLOR))
    fig.add_trace(go.Bar(name=f"SWA {swa_method[3:]}", x=models, y=swa, marker_color=SWA_COLOR))
    fig.add_trace(go.Bar(
        name="Best linearized", x=models, y=[None if r is None else r["avg"] for r in best], marker_color=LINEAR_COLOR,
        hovertext=["not reported" if r is None else _short_name(r["method"]) for r in best],
        hovertemplate="%{x}<br>%{hovertext}: %{y}<extra>Best linearized</extra>",
    ))
    fig.update_layout(barmode="group", height=360, margin=dict(l=10, r=10, t=40, b=10), yaxis=dict(title="Average score (%)", range=y_range),
                      legend=dict(orientation="h", y=-0.35), title=title)
    return fig


def _how_to_read_bars(table: list[dict], window: int) -> None:
    """Explain the three bars of the grouped chart, naming the best linearized method of each base model."""
    best = {m: _best_linear_row([r for r in table if r["model"] == m], "avg") for m in _models(table)}
    names = "; ".join(f"{m}: {'none reported' if r is None else _short_name(r['method'])}" for m, r in best.items())
    st.caption(
        "**How to read the bars.** "
        "**Teacher** = the original pretrained model with ordinary full attention: the score to match (100% recovery). "
        f"**SWA({window}, 4)** = that same model with its attention restricted to a sliding window of {window} tokens plus 4 sink tokens, with no training at all. "
        "**Best linearized** = for each base model, the highest score among the retrofitted linear-attention variants reported for it, i.e. the strongest competitor; "
        f"the method can differ from model to model (hover a bar to see it): {names}."
    )


# ----------------------------------------------------------------------------------------
# 1. Setup and metrics
# ----------------------------------------------------------------------------------------
def _render_setup() -> None:
    st.subheader("The question: is a plain sliding window enough, or must we retrofit linear attention?")
    st.markdown(
        "Full attention costs $O(L^2)$ and its **KV cache** (stored keys and values) grows with every token. Two ways to cap the cost: "
        "**retrofit** a pretrained LLM to *linear attention* (needs extra training), or use the simplest KV-cache eviction, "
        "**Sliding Window Attention (SWA) with attention sinks**. Earlier linearization papers compared only against SWA *without* sinks, "
        "which fails catastrophically. This paper adds the **missing comparison**."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        with st.container(border=True):
            st.caption("The proposed baseline")
            st.markdown("**SWA with sinks, $\\text{SWA}(w, s)$**")
            st.markdown("Each token attends to the first $s = 4$ tokens (the *sinks*) and the most recent $w - 4$ tokens. **Training-free**, memory capped at $w$ tokens.")
    with c2:
        with st.container(border=True):
            st.caption("The competition")
            st.markdown("**Retrofitted linear attention**")
            st.markdown("Swap softmax attention for a linear form with a fixed-size state, then train. Examples: LoLCATs (40M tokens), Liger-GLA, QRWKV, Llamba.")
    with c3:
        with st.container(border=True):
            st.caption("How results are reported")
            st.markdown("**Recovery %**")
            st.markdown("**Teacher** = the original full-attention model. **Recovery** = student score divided by teacher score, so 100% means no loss. Models: 1.3B to 72B.")

    st.markdown("#### The benchmarks")
    md_table(
        ["Benchmark", "What it tests", "Metric"],
        [
            ("MMLU, ARC, HellaSwag, PIQA, WinoGrande", "Knowledge, science questions, commonsense, pronoun resolution (the Winograd task of our Code section); short multiple-choice prompts", "accuracy"),
            ("S-NIAH", "Single *needle in a haystack*: retrieve one fact hidden in long filler text", "accuracy"),
            ("BABILong", "Reasoning over facts hidden inside long book text", "accuracy"),
            ("LongBench, RULER", "Long-context understanding and synthetic retrieval / tracking / counting tasks", "accuracy"),
        ],
    )


# ----------------------------------------------------------------------------------------
# 2. Short context
# ----------------------------------------------------------------------------------------
def _render_short() -> None:
    st.subheader("Short-context knowledge and reasoning")
    swa = next(r for r in TABLE_1 if r["method"].startswith("SWA"))
    models = _models(TABLE_2)
    wins = 0
    for m in models:
        rows = [r for r in TABLE_2 if r["model"] == m]
        best = _best_linear(rows, "avg")
        wins += best is None or next(r for r in rows if r["kind"] == "SWA")["avg"] >= best

    k1, k2, k3 = st.columns(3)
    k1.metric("SWA(64, 4): average recovery", f"{swa['avg']}%", help="Student score / teacher score on the average of 6 benchmarks")
    k2.metric("Retrofitting tokens needed by SWA", "0", help="LoLCATs: 40M, Mamba in the Llama: 20B, SUPRA: 100B")
    k3.metric("Best average score", f"{wins} of {len(models)} models", help="SWA vs the best linearized method for each base model")

    md_table(
        ["Method", "Retrofit tokens", "MMLU recovery %", "Average recovery %"],
        [
            (f"**{r['method']}**" if r["method"].startswith("SWA") else r["method"], r["tokens"], r["mmlu"], "-" if r["avg"] is None else r["avg"])
            for r in TABLE_1
        ],
    )
    st.plotly_chart(_grouped_average_chart(TABLE_2, "SWA(64,4)", "Average of 6 benchmarks per base model (axis starts at 40)", [40, 85]), width="stretch")
    _how_to_read_bars(TABLE_2, 64)


# ----------------------------------------------------------------------------------------
# 4. Long context
# ----------------------------------------------------------------------------------------
def _render_long() -> None:
    st.subheader("Long-context reasoning")

    st.markdown("#### Single needle in a haystack (Llama-3.1-8B)")
    c1, c2 = st.columns(2)
    task = c1.radio("S-NIAH variant", ["S-NIAH-1", "S-NIAH-2", "S-NIAH-3"], horizontal=True, index=2, key="res_swa_task")
    window = c2.radio("Window size", ["128", "256", "512"], horizontal=True, index=1, key="res_swa_window")
    fa = next(r for r in TABLE_4 if r["window"] == "inf")["scores"][task]
    colors = {"SWA with sinks": SWA_COLOR, "LoLCATs": LINEAR_COLOR, "Liger-GLA": GREEN}
    fig = go.Figure()
    recovery = {}
    for r in (r for r in TABLE_4 if r["window"] == window):
        label = _method_label(r["method"])
        fig.add_trace(go.Bar(name=label, x=CONTEXTS_S_NIAH, y=r["scores"][task], marker_color=colors[label]))
        recovery[label] = 100 * r["scores"][task][3] / fa[3]
    fig.add_trace(go.Scatter(name="Full attention", x=CONTEXTS_S_NIAH, y=fa, mode="lines+markers", line=dict(color=TEACHER_COLOR, dash="dash")))
    fig.update_layout(barmode="group", height=340, margin=dict(l=10, r=10, t=30, b=10), xaxis_title="Context length", yaxis=dict(title="Accuracy (%)", range=[0, 105]),
                      legend=dict(orientation="h", y=-0.3), title=f"{task}, window {window}")
    st.plotly_chart(fig, width="stretch")
    m1, m2 = st.columns(2)
    m1.metric("SWA: share of full-attention accuracy at 4K", f"{recovery['SWA with sinks']:.1f}%")
    m2.metric("LoLCATs: share at 4K", f"{recovery['LoLCATs']:.1f}%")

    st.markdown("#### BABILong, window 256 (Llama-3.1-8B)")
    contexts = ["0K", "1K", "2K", "4K"]
    fig = go.Figure()
    line_colors = {**colors, "Full attention": TEACHER_COLOR}
    for r in TABLE_5:
        label = _method_label(r["method"])
        fig.add_trace(go.Scatter(name=label, x=contexts, y=r["scores"], mode="lines+markers", line=dict(color=line_colors[label], width=3)))
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=10), xaxis_title="Context length (0K = no distractor text)", yaxis=dict(title="Accuracy (%)", range=[0, 80]),
                      legend=dict(orientation="h", y=-0.3))
    st.plotly_chart(fig, width="stretch")
    st.caption("LoLCATs is slightly ahead at short context and falls away as the context grows.")

    st.markdown("#### Longer contexts, up to 16K tokens, five base models")
    st.plotly_chart(_grouped_average_chart(TABLE_6, "SWA(512,4)", "Average of LongBench, LongBench-v2, RULER and BABILong"), width="stretch")
    _how_to_read_bars(TABLE_6, 512)
    st.caption("Full-attention, LoLCATs and Liger-GLA baselines for the first two charts are taken from Meng et al. (2026). All fixed-memory methods remain far from full attention, especially on RULER.")


# ----------------------------------------------------------------------------------------
# 5. Speed, memory and take-aways
# ----------------------------------------------------------------------------------------
def _render_verdict() -> None:
    st.subheader("Speed, memory and take-aways")
    md_table(
        ["Attention type", "Decoding speed as context grows", "KV cache / state memory"],
        [
            ("Full attention", "Drops once the context passes about 1K tokens", "Grows linearly with context"),
            ("**SWA with sinks**", "**Flat and the fastest** (window 64 faster than 512)", "Grows until the window is full, then **constant**; window 64 is the lowest of all"),
            ("Linear attention", "Flat", "Constant, second lowest after SWA window 64"),
            ("Linear + SWA (LoLCATs)", "Flat, slower than SWA", "Constant, above Linear"),
        ],
    )
    st.caption("Measured on a 4-layer Transformer with contexts from 128 to 256K tokens. The authors give these results as plots, so only the ranking is summarised here.")

    c1, c2, c3 = st.columns(3)
    for col, title, body in (
        (c1, "Short context", "SWA(64, 4) recovers 99% of the average score with **zero** training, level with the best retrofits."),
        (c2, "Long context", "SWA beats retrofitted linear attention by 2 to 10 times on needle and BABILong tasks."),
        (c3, "Efficiency", "Fastest decoding and a small fixed memory, with no special kernels or post-training."),
    ):
        with col:
            with st.container(border=True):
                st.markdown(f"**{title}**")
                st.markdown(body)

    st.markdown(
        "**Limits the authors state:** only training-free SWA is studied (post-training could help); only the simplest eviction rule "
        "(score-based methods may do better); no hybrid models with some full-attention layers."
    )
    st.info("SWA is the *restricted self-attention* idea from the Transformer paper's complexity analysis, tested at scale.")


# ----------------------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------------------
def render_swa_paper():
    tabs = st.tabs(["Setup & Metrics", "Short Context", "Long Context", "Speed & Take-aways"])
    for tab, render in zip(tabs, [_render_setup, _render_short, _render_long, _render_verdict]):
        with tab:
            render()
