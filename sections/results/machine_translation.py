"""
========================================================================================
Module: Results - Machine Translation (Table 2)
Author: Gusev Roman (Results Interpreter & Challenger)
========================================================================================
Paper Sec. 6.1: BLEU on WMT 2014 newstest2014 and estimated training cost in FLOPs.
========================================================================================
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from sections.results.common import challenger_box, md_table, speaker_notes
from sections.results.paper_data import AMBER, DARK, GPU_TFLOPS, SLATE, table_2_df

KIND_STYLE = {
    "single": ("Single model (previous work)", SLATE),
    "ensemble": ("Ensemble (previous work)", DARK),
    "transformer": ("Transformer (this paper)", AMBER),
}
LANGS = {"English-to-German": ("BLEU EN-DE", "FLOPs EN-DE"), "English-to-French": ("BLEU EN-FR", "FLOPs EN-FR")}


def _fmt_flops(value) -> str:
    """FLOPs as LaTeX (for markdown text and tables), e.g. $2.3 \\cdot 10^{19}$."""
    if pd.isna(value):
        return "-"
    mantissa, exponent = f"{value:.1e}".split("e")
    return f"${float(mantissa):.1f} \\cdot 10^{{{int(exponent)}}}$"


def _fmt_flops_plain(value: float) -> str:
    """FLOPs as plain text for st.metric values, which cannot render LaTeX."""
    return f"{value:.2e}"


def _fmt_bleu(value) -> str:
    return "-" if pd.isna(value) else f"{value:g}"


# ----------------------------------------------------------------------------------------
# 1. The table and the BLEU comparison
# ----------------------------------------------------------------------------------------
def _render_table_and_bleu(df: pd.DataFrame) -> None:
    st.subheader("New State-of-The-Art on both translation tasks")
    st.markdown(
        "Every system is scored on **newstest2014** (the WMT 2014 test set) with **BLEU**, and its estimated "
        "**training cost** is given in FLOPs. '-' means the original authors did not report the value. "
        "*Single model* = one trained network; *ensemble* = several networks whose outputs are averaged "
        "(better, but several times more training and inference cost)."
    )

    shown = df[["Model", "BLEU EN-DE", "BLEU EN-FR", "FLOPs EN-DE", "FLOPs EN-FR"]].copy()
    for col in ("BLEU EN-DE", "BLEU EN-FR"):
        shown[col] = shown[col].map(_fmt_bleu)
    for col in ("FLOPs EN-DE", "FLOPs EN-FR"):
        shown[col] = shown[col].map(_fmt_flops)
    md_table(list(shown.columns), list(shown.itertuples(index=False, name=None)))
    st.caption("The paper prints a single cost cell for each Transformer row; it is shown under both languages. Source: Vaswani et al. 2017, Table 2.")

    lang = st.radio("Language pair", list(LANGS), horizontal=True, key="res_mt_lang")
    bleu_col, _ = LANGS[lang]
    show_ens = st.checkbox("Include ensembles", value=True, key="res_mt_ens")

    data = df.dropna(subset=[bleu_col])
    if not show_ens:
        data = data[data["Kind"] != "ensemble"]
    data = data.sort_values(bleu_col)

    fig = go.Figure()
    for kind, (name, color) in KIND_STYLE.items():
        part = data[data["Kind"] == kind]
        if part.empty:
            continue
        fig.add_trace(go.Bar(
            y=part["Model"], x=part[bleu_col], orientation="h", name=name, marker_color=color,
            text=[f"{v:g}" for v in part[bleu_col]], textposition="outside", cliponaxis=False,
        ))
    low = data[bleu_col].min()
    fig.update_layout(
        height=120 + 42 * len(data), margin=dict(l=10, r=40, t=30, b=10),
        xaxis=dict(title="BLEU on newstest2014 (higher is better)", range=[max(0, low - 3), data[bleu_col].max() + 1.5]),
        legend=dict(orientation="h", y=-0.25), title=f"{lang}: BLEU (axis does not start at 0, differences are small)",
    )
    st.plotly_chart(fig, width="stretch")

    # Headline deltas, computed from the table rather than typed by hand
    prev = df[df["Kind"] != "transformer"]
    base_row = df[df["Model"] == "Transformer (base model)"].iloc[0]
    big_row = df[df["Model"] == "Transformer (big)"].iloc[0]
    best_prev = prev[bleu_col].max()
    best_prev_name = prev.loc[prev[bleu_col].idxmax(), "Model"]
    m1, m2, m3 = st.columns(3)
    m1.metric("Best previous system", f"{best_prev:g} BLEU", help=best_prev_name)
    m2.metric("Transformer (base)", f"{base_row[bleu_col]:g} BLEU", delta=f"{base_row[bleu_col] - best_prev:+.2f}")
    m3.metric("Transformer (big)", f"{big_row[bleu_col]:g} BLEU", delta=f"{big_row[bleu_col] - best_prev:+.2f}")

    # speaker_notes(
    #     "- Read the table left to right: model, BLEU on German, BLEU on French, then cost.\n"
    #     "- EN-DE is the clean win: base 27.3 and big 28.4 vs. 26.36 for the best earlier ensemble.\n"
    #     "- Toggle to French: base is *worse* than all the old systems (38.1), big wins (41.8). Say it plainly, don't hide it.\n"
    #     "- Mention the 41.0 vs 41.8 discrepancy between text and table if asked about EN-FR.\n"
    #     "- Bridge: *\"BLEU is only half the claim. The other half is cost.\"*"
    # )


# ----------------------------------------------------------------------------------------
# 2. Quality vs cost
# ----------------------------------------------------------------------------------------
def _render_quality_vs_cost(df: pd.DataFrame) -> None:
    st.subheader("The real claim: better *and* cheaper")
    st.markdown(
        "Table 2's caption says the Transformer wins 'at a fraction of the training cost'. "
        "Plot **BLEU against training cost** (log scale): the best systems sit **top-left** (high quality, low cost)."
    )
    lang = st.radio("Language pair", list(LANGS), horizontal=True, key="res_cost_lang")
    bleu_col, flops_col = LANGS[lang]
    data = df.dropna(subset=[bleu_col, flops_col])

    fig = go.Figure()
    for kind, (name, color) in KIND_STYLE.items():
        part = data[data["Kind"] == kind]
        if part.empty:
            continue
        fig.add_trace(go.Scatter(
            x=part[flops_col], y=part[bleu_col], mode="markers+text", name=name, text=part["Model"], textposition="top center",
            marker=dict(size=16, color=color, symbol="diamond" if kind == "ensemble" else "circle"),
            hovertemplate="%{text}<br>BLEU %{y}<br>%{x:.1e} FLOPs<extra></extra>",
        ))
    fig.update_layout(
        height=480, margin=dict(l=10, r=10, t=30, b=10), xaxis_type="log",
        xaxis_title="Estimated training cost (FLOPs, log scale)", yaxis_title="BLEU", legend=dict(orientation="h", y=-0.2),
    )
    st.plotly_chart(fig, width="stretch")

    prev = data[data["Kind"] != "transformer"]
    big = data[data["Model"] == "Transformer (big)"].iloc[0]
    base = data[data["Model"] == "Transformer (base model)"].iloc[0]
    best_single = prev[prev["Kind"] == "single"].sort_values(bleu_col).iloc[-1]
    best_ens = prev[prev["Kind"] == "ensemble"].sort_values(bleu_col).iloc[-1]

    c1, c2, c3 = st.columns(3)
    c1.metric(f"Big vs best single ({best_single['Model']})", f"{best_single[flops_col] / big[flops_col]:.1f}x cheaper" if best_single[flops_col] > big[flops_col] else f"{big[flops_col] / best_single[flops_col]:.1f}x more expensive")
    c2.metric(f"Big vs best ensemble ({best_ens['Model']})", f"{best_ens[flops_col] / big[flops_col]:.1f}x cheaper")
    c3.metric("Base vs big", f"{big[flops_col] / base[flops_col]:.1f}x cheaper", delta=f"{base[bleu_col] - big[bleu_col]:+.1f} BLEU", delta_color="off")

    st.markdown("#### Where do the FLOPs numbers come from?")
    st.markdown(
        "The authors did **not** count operations. They estimated: **training time $\\times$ number of GPUs $\\times$ sustained single-precision TFLOPS per GPU** "
        "(footnote 5). Try to reproduce the Transformer rows:"
    )
    preset = st.radio("Preset", ["Transformer base (12 h)", "Transformer big (3.5 days)", "Custom"], horizontal=True, key="res_flops_preset")
    defaults = {"Transformer base (12 h)": 12.0, "Transformer big (3.5 days)": 84.0}
    c1, c2, c3 = st.columns(3)
    hours = c1.number_input("Training time (hours)", min_value=0.1, value=defaults.get(preset, 24.0), disabled=preset != "Custom", key=f"res_flops_h_{preset}")
    gpus = c2.number_input("Number of GPUs", min_value=1, value=8, key="res_flops_g")
    gpu = c3.selectbox("GPU", list(GPU_TFLOPS), index=3, format_func=lambda g: f"{g} ({GPU_TFLOPS[g]} TFLOPS)", key="res_flops_t")
    flops = hours * 3600 * gpus * GPU_TFLOPS[gpu] * 1e12
    expected = {"Transformer base (12 h)": 3.3e18, "Transformer big (3.5 days)": 2.3e19}.get(preset)
    st.metric("Estimated training FLOPs", _fmt_flops_plain(flops), help="time (s) $\\times$ GPUs $\\times$ TFLOPS $\\times 10^{12}$")
    if expected:
        st.caption(f"Paper (Table 2): {_fmt_flops(expected)}. The formula reproduces it, so the 'cost' is a *wall-clock $\\times$ peak-throughput* estimate, not a profiler measurement.")

# ----------------------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------------------
def render_machine_translation():
    st.header("Machine Translation Results")
    st.caption("WMT 2014 English-to-German and English-to-French: translation quality (BLEU) and estimated training cost (FLOPs).")

    df = table_2_df()
    tabs = st.tabs(["📋 BLEU Score Comparison", "💰 Quality vs Cost"])
    with tabs[0]:
        _render_table_and_bleu(df)
    with tabs[1]:
        _render_quality_vs_cost(df)
