"""
========================================================================================
Module: Results - Model Variations / Ablations (Table 3)
Author: Gusev Roman (Results Interpreter & Challenger)
========================================================================================
Paper Sec. 6.2: change one thing at a time in the base model and measure EN-DE dev quality.
========================================================================================
"""

import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from sections.results.common import challenger_box, md_table, speaker_notes
from sections.results.paper_data import AMBER, BLUE, GREEN, GROUP_TITLES, SLATE, table_3_df

GROUP_COLORS = {"base": AMBER, "A": BLUE, "B": GREEN, "C": "#a78bfa", "D": "#f472b6", "E": SLATE, "big": AMBER}

# Short definitions of the Table 3 columns, following the paper's Sec. 3 (architecture) and Sec. 5 (training)
PARAMETER_GLOSSARY = [
    ("$N$", "Number of identical layers stacked in the encoder and in the decoder. Each layer = an attention block + a feed-forward block.", "6"),
    ("$d_{\\text{model}}$", "Width of the model: the size of every token vector, in the embeddings and at the output of every sub-layer.", "512"),
    ("$d_{ff}$", "Hidden width of the feed-forward block (two linear layers with a ReLU between, applied to each position separately).", "2048"),
    ("$h$", "Number of attention heads: parallel attention operations, each on its own learned projection of the input.", "8"),
    ("$d_k$", "Size of the query and key vectors of one head (the dot product $q \\cdot k$ sums over $d_k$ numbers).", "64"),
    ("$d_v$", "Size of the value vector of one head, i.e. how much information each head passes on.", "64"),
    ("$P_{drop}$", "Dropout rate: the share of values randomly set to zero during training, which fights over-fitting.", "0.1"),
    ("$\\epsilon_{ls}$", "Label smoothing: the correct word gets probability $1-\\epsilon_{ls}$ instead of 1, the rest is spread over other words.", "0.1"),
    ("train steps", "Number of parameter updates; one step processes one batch of about 25,000 source and 25,000 target tokens.", "100K"),
    ("PPL, BLEU, params", "Results: perplexity per word-piece (lower is better), BLEU on the dev set (higher is better), and parameters in millions.", "4.92, 25.8, 65"),
]

# What the paper says about each group (Sec. 6.2) and what we add as interpretation
READING = {
    "A": (
        "Computation is held constant: more heads means a smaller $d_k$ = $d_v$ = $d_{\\text{model}}$ / h per head. "
        "**One head is 0.9 BLEU worse** than the best setting, and **too many heads also hurts** (h = 32 drops to 25.4). "
        "The sweet spot is a moderate number of heads (8-16).",
        "Interpretation: heads let the model attend to different relations at once (see the BertViz tab), "
        "but each head needs enough dimensions to be useful: at h = 32 each head has only 16.",
    ),
    "B": (
        "Shrinking the key size $d_k$ **hurts quality** (25.8 $\\rightarrow$ 25.4 $\\rightarrow$ 25.1). "
        "The authors read this as: *'determining compatibility is not easy'*, and a more sophisticated compatibility function than the dot product may help.",
        "Interpretation: this is a hint, not a proof. The ablation also removes parameters (65 M $\\rightarrow$ 58 M), so size and key-width are confounded.",
    ),
    "C": (
        "**Bigger is better**: 2 layers 23.7 $\\rightarrow$ 4 layers 25.3 $\\rightarrow$ 6 layers 25.8 $\\rightarrow$ 8 layers 25.5 (BLEU peaks at 6); "
        "$d_{\\text{model}}$ 256 $\\rightarrow$ 512 $\\rightarrow$ 1024 gives 24.5 $\\rightarrow$ 25.8 $\\rightarrow$ 26.0; $d_{ff}$ 1024 $\\rightarrow$ 2048 $\\rightarrow$ 4096 gives 25.4 $\\rightarrow$ 25.8 $\\rightarrow$ 26.2. "
        "Perplexity falls steadily (6.11 $\\rightarrow$ 4.88 with depth, 5.75 $\\rightarrow$ 4.66 with width).",
        "Interpretation: returns diminish quickly. 168 M parameters ($d_{\\text{model}}$ 1024) buy only +0.2 BLEU over the 65 M base, and 8 layers is slightly *worse* than 6 in BLEU "
        "although perplexity still improves. The big model combines the winners of this group.",
    ),
    "D": (
        "**Dropout is very helpful against over-fitting**: 0.0 $\\rightarrow$ 24.6 BLEU, 0.1 $\\rightarrow$ 25.8, 0.2 $\\rightarrow$ 25.5. "
        "Label smoothing shows the metric conflict: $\\epsilon$ = 0.0 gives the *best perplexity* (4.67) but *lower BLEU* (25.3); $\\epsilon$ = 0.1 gives worse perplexity (4.92) and the best BLEU (25.8).",
        "Interpretation: perplexity rewards confident predictions, BLEU rewards good *translations*. A model that is deliberately less sure translates better, "
        "which is why the paper keeps $\\epsilon$ = 0.1 despite the worse perplexity.",
    ),
    "E": (
        "Replacing the fixed **sinusoidal** positional encoding with **learned** positional embeddings gives *nearly identical* results (25.7 vs 25.8 BLEU, same perplexity 4.92).",
        "Interpretation: the model does not need the learned version. The sinusoids were chosen partly because they might extrapolate to sequences longer than those seen in training, "
        "a claim Table 3 itself cannot test.",
    ),
}


def _render_table_and_chart(df) -> None:
    st.subheader("Which parts of the Transformer matter?")
    st.markdown(
        "An **ablation study** changes one component at a time and measures what breaks. Starting from the base model "
        "(N = 6 layers, $d_{\\text{model}}$ = 512, $d_{ff}$ = 2048, h = 8 heads, $d_k$ = $d_v$ = 64, dropout 0.1, label smoothing 0.1, 100K steps), "
        "the authors vary rows (A)-(E) and report **perplexity** and **BLEU** on the **EN-DE development set (newstest2013)**."
    )

    st.markdown("#### What each column of Table 3 means")
    md_table(["Parameter", "What it controls", "Base value"], PARAMETER_GLOSSARY)

    options = ["A", "B", "C", "D", "E"]
    group = st.radio("Experiment group", options, format_func=lambda g: GROUP_TITLES[g], horizontal=False, key="res_var_group")

    base = df[df["Group"] == "base"].iloc[0]
    part = df[df["Group"].isin([group, "base"])]
    part = part.copy()
    part["Variation"] = part["Variation"].where(part["Group"] != "base", "base (reference)")
    colors = [AMBER if g == "base" else GROUP_COLORS[group] for g in part["Group"]]

    fig = make_subplots(rows=1, cols=2, subplot_titles=("BLEU on dev (higher is better)", "Perplexity on dev (lower is better)"))
    fig.add_trace(go.Bar(x=part["Variation"], y=part["BLEU (dev)"], marker_color=colors, text=part["BLEU (dev)"], textposition="outside", cliponaxis=False), row=1, col=1)
    fig.add_trace(go.Bar(x=part["Variation"], y=part["PPL (dev)"], marker_color=colors, text=part["PPL (dev)"], textposition="outside", cliponaxis=False), row=1, col=2)
    fig.add_hline(y=base["BLEU (dev)"], line_dash="dash", line_color=SLATE, row=1, col=1)
    fig.add_hline(y=base["PPL (dev)"], line_dash="dash", line_color=SLATE, row=1, col=2)
    bleu_lo, bleu_hi = part["BLEU (dev)"].min(), part["BLEU (dev)"].max()
    ppl_lo, ppl_hi = part["PPL (dev)"].min(), part["PPL (dev)"].max()
    fig.update_yaxes(range=[bleu_lo - 1.0, bleu_hi + 0.6], row=1, col=1)
    fig.update_yaxes(range=[max(0, ppl_lo - 0.8), ppl_hi + 0.5], row=1, col=2)
    fig.update_layout(height=400, showlegend=False, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, width="stretch")
    st.caption("Dashed line = base model. Both axes are zoomed (they do not start at 0) because the differences are about 1 BLEU.")

    paper_says, our_reading = READING[group]
    c1, c2 = st.columns(2)
    with c1:
        with st.container(border=True):
            st.markdown("**What the paper reports**")
            st.markdown(paper_says)
    with c2:
        with st.container(border=True):
            st.markdown("**How to interpret it**")
            st.markdown(our_reading)

    st.markdown("#### Base versus big")
    big = df[df["Group"] == "big"].iloc[0]
    c1, c2, c3 = st.columns(3)
    c1.metric("Parameters: base to big", f"{base['Params (M)']} M to {big['Params (M)']} M", delta=f"{big['Params (M)'] / base['Params (M)']:.1f}x", delta_color="off")
    c2.metric("Dev BLEU: base to big", f"{base['BLEU (dev)']} to {big['BLEU (dev)']}", delta=f"{big['BLEU (dev)'] - base['BLEU (dev)']:+.1f}")
    c3.metric("Dev perplexity: base to big", f"{base['PPL (dev)']} to {big['PPL (dev)']}", delta=f"{big['PPL (dev)'] - base['PPL (dev)']:+.2f}", delta_color="inverse")

    st.markdown("#### Take-aways")
    t1, t2, t3 = st.columns(3)
    with t1:
        with st.container(border=True):
            st.markdown("**Biggest losses**")
            st.markdown("- 2 layers: -2.1 BLEU\n- no dropout: -1.2\n- 1 head: -0.9\n- $d_{\\text{model}}$ = 256: -1.3")
    with t2:
        with st.container(border=True):
            st.markdown("**Biggest gains**")
            st.markdown("- big model: +0.6 (213 M params)\n- $d_{ff}$ = 4096: +0.4\n- $d_{\\text{model}}$ = 1024: +0.2\n- (the base model is already near a plateau)")
    with t3:
        with st.container(border=True):
            st.markdown("**No difference**")
            st.markdown("- learned vs sinusoidal positions: -0.1\n- 16 vs 8 heads: 0.0\n- $\\epsilon_{ls}$ = 0.2: -0.1")


def render_model_variations():
    df = table_3_df()
    _render_table_and_chart(df)
