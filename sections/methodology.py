"""
========================================================================================
Module: Methodology
Authors: Leahu Teodor Filip & Szocs Luca (Methodology Experts)
========================================================================================
This file is fully isolated. Any changes made here appear in the
'Methodology' tab without affecting the other sections.

Proposed split (edit PRESENTERS below if you change it):
  Part A - The model (sections 3.1-3.5, Figures 1-5):           Teodor
  Part B - Training & experiments (sections 5-6, Tables 2-4):   Luca
Every number on these slides comes from Vaswani et al. (2017) unless a caption says
otherwise. Simulations are labelled as such.
========================================================================================
"""

import html
import math
from collections import Counter

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

PRESENTERS = {"A": "Teodor", "B": "Luca"}

BLUE, AMBER, GREEN, SLATE, INDIGO, ROSE = "#38bdf8", "#f59e0b", "#10b981", "#64748b", "#6366f1", "#f43f5e"
HEAD_COLORS = [BLUE, AMBER, GREEN, INDIGO, ROSE, "#a855f7", "#14b8a6", "#eab308"]

# Base model hyper-parameters (Table 3, row "base")
LEGEND = dict(orientation="h", yanchor="top", y=-0.28)
BASE = dict(N=6, d_model=512, d_ff=2048, h=8, d_k=64, d_v=64, p_drop=0.1, eps_ls=0.1, vocab=37000)


# ----------------------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------------------
def _speaker_notes(text: str) -> None:
    with st.expander("🎤 Speaker notes"):
        st.markdown(text)


def _owner(part: str) -> None:
    st.caption(f"Part {part} · presented by **{PRESENTERS[part]}**")


def _box(title: str, body: str, color: str = SLATE, bold: bool = False) -> str:
    border = f"2px solid {color}" if bold else f"1px solid {color}"
    bg = f"{color}33" if bold else "transparent"
    return (
        f'<div style="border:{border};background:{bg};border-radius:10px;padding:6px 10px;margin:4px 0;'
        f'text-align:center;font-size:0.92rem;"><b>{title}</b>'
        + (f'<div style="font-size:0.78rem;opacity:0.8;">{body}</div>' if body else "")
        + "</div>"
    )


def _chips(items: list[str], color: str = BLUE) -> str:
    parts = [
        f'<span style="border:1px solid {color};padding:4px 10px;margin:3px;border-radius:8px;'
        f'display:inline-block;font-size:0.95rem;">{html.escape(t)}</span>'
        for t in items
    ]
    return '<div style="line-height:2.4;">' + '<span style="opacity:0.6;"> → </span>'.join(parts) + "</div>"


def _tok(text: str, fallback: str, limit: int = 14) -> list[str]:
    toks = text.split()[:limit]
    return toks if toks else fallback.split()


# ----------------------------------------------------------------------------------------
# ⓪ Overview: what the methodology consists of and where every figure/table comes from
# ----------------------------------------------------------------------------------------
PROVENANCE = [
    ("Figure 1", "Architecture diagram", "§3", "Drawn by the authors; no data", "A · ①"),
    ("Figure 2", "Scaled dot-product & multi-head attention", "§3.2", "Drawn by the authors; no data", "A · ②"),
    ("Table 1", "Cost of self-attention vs RNN vs CNN layers", "§4", "Analytical big-O, not measured", "Problem Framing"),
    ("Table 2", "BLEU on newstest2014 + training cost", "§6.1", "Test set; checkpoint-averaged base & big models; cost = time × GPUs × TFLOPS; other rows copied from earlier papers", "B · ⑧ (setup) / Results"),
    ("Table 3", "Model variations (ablations)", "§6.2", "Dev set newstest2013; EN→DE; no checkpoint averaging; one change at a time vs base", "B · ⑨"),
    ("Table 4", "English constituency parsing F1", "§6.3", "WSJ Section 23; 4-layer model; WSJ-only and semi-supervised", "B · ⑩ (setup) / Results"),
    ("Figures 3–5", "Attention weight visualisations", "Appendix", "Encoder self-attention, layer 5 of 6, trained EN→DE model, one sentence each", "A · ④"),
]

PIPELINE = [
    "WMT 2014 text",
    "BPE / word-pieces",
    "Batches of ~25k tokens",
    "Transformer (base / big)",
    "Adam + warm-up, dropout, label smoothing",
    "8 × P100 · 12 h / 3.5 days",
    "Checkpoint averaging",
    "Beam search (k=4, α=0.6)",
    "BLEU",
]


def _render_overview() -> None:
    st.subheader("What 'methodology' means in this paper")
    st.markdown(
        "The paper has **one method** (the Transformer, §3) and **three experiments**: "
        "machine translation (§6.1), an ablation study (§6.2) and English constituency parsing (§6.3). "
        "Everything between them is the training recipe (§5)."
    )
    c1, c2 = st.columns(2)
    with c1:
        with st.container(border=True):
            st.markdown(f"**Part A · The model** — {PRESENTERS['A']}")
            st.markdown("① Architecture (Fig. 1)  \n② Attention, three ways (Fig. 2)  \n③ Word order, embeddings, FFN  \n④ Reading the attention figures (Figs. 3–5)")
    with c2:
        with st.container(border=True):
            st.markdown(f"**Part B · Training & experiments** — {PRESENTERS['B']}")
            st.markdown("⑤ Data & batching  \n⑥ Optimizer, schedule, hardware  \n⑦ Regularization  \n⑧ Decoding & evaluation protocol  \n⑨ Model variations (Table 3)  \n⑩ Parsing setup + the 2026 paper's setup")

    st.markdown("#### The experimental pipeline, end to end")
    st.markdown(_chips(PIPELINE), unsafe_allow_html=True)

    st.markdown("#### Where every figure and table comes from")
    df = pd.DataFrame(PROVENANCE, columns=["Item", "Shows", "Section", "How it was produced", "Covered in"])
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.caption("A methodology expert should be able to answer, for each row: which data, which model, which split, which metric.")

    with st.expander("📚 Sources used in this section"):
        st.markdown(
            "- Vaswani et al. (2017). *Attention Is All You Need.* [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)\n"
            "- Jay Alammar. *The Illustrated Transformer.* [jalammar.github.io](https://jalammar.github.io/illustrated-transformer/)\n"
            "- Sebastian Raschka. *Understanding Encoder and Decoder LLMs.* [magazine.sebastianraschka.com](https://magazine.sebastianraschka.com/p/understanding-encoder-and-decoder)\n"
            "- Sennrich et al. (2016). *Neural Machine Translation of Rare Words with Subword Units* (BPE). [arXiv:1508.07909](https://arxiv.org/abs/1508.07909)\n"
            "- Szegedy et al. (2016). *Rethinking the Inception Architecture* (label smoothing). [arXiv:1512.00567](https://arxiv.org/abs/1512.00567)\n"
            "- Wu et al. (2016). *Google's NMT System* (length penalty). [arXiv:1609.08144](https://arxiv.org/abs/1609.08144)\n"
            "- Press & Wolf (2017). *Using the Output Embedding to Improve Language Models* (weight tying). [arXiv:1608.05859](https://arxiv.org/abs/1608.05859)\n"
            "- Xiong et al. (2020). *On Layer Normalization in the Transformer Architecture.* [arXiv:2002.04745](https://arxiv.org/abs/2002.04745)\n"
            "- Jain & Wallace (2019). *Attention is not Explanation.* [arXiv:1902.10186](https://arxiv.org/abs/1902.10186); Wiegreffe & Pinter (2019). *Attention is not not Explanation.* [arXiv:1908.04626](https://arxiv.org/abs/1908.04626)\n"
            "- Jolicoeur-Martineau et al. (2026). *Sliding-Window Beats Linear Attention.* [arXiv:2608.28444](https://arxiv.org/abs/2608.28444)"
        )

    _speaker_notes(
        "- Frame the section: one model, a training recipe, three experiments.\n"
        "- Walk the pipeline chips left to right; each chip is one sub-tab.\n"
        "- The provenance table is the key slide for a methodology expert: Table 2 is on the **test** set with checkpoint averaging, "
        "Table 3 is on the **dev** set without it. Numbers from the two tables are not directly comparable.\n"
        f"- Hand over: *\"{PRESENTERS['A']} starts with the model itself.\"*"
    )


# ----------------------------------------------------------------------------------------
# ① Architecture (Figure 1)
# ----------------------------------------------------------------------------------------
COMPONENTS = {
    "in_emb": "Input embedding",
    "pe": "Positional encoding",
    "enc_mha": "Encoder self-attention",
    "addnorm": "Add & Norm",
    "ffn": "Feed-forward",
    "out_emb": "Output embedding (shifted right)",
    "dec_mha": "Masked decoder self-attention",
    "cross": "Encoder–decoder attention",
    "head": "Linear + softmax",
}


def _arch_html(sel: str) -> str:
    def b(key, title, body="", color=SLATE):
        return _box(title, body, AMBER if key == sel else color, bold=(key == sel))

    enc = (
        '<div style="border:1px dashed #64748b;border-radius:12px;padding:8px;">'
        '<div style="font-size:0.75rem;opacity:0.7;text-align:right;">N = 6 ×</div>'
        + b("addnorm", "Add & Norm")
        + b("ffn", "Feed Forward", "512 → 2048 → 512", GREEN)
        + b("addnorm", "Add & Norm")
        + b("enc_mha", "Multi-Head Attention", "self, all positions", BLUE)
        + "</div>"
        + b("pe", "⊕ Positional Encoding")
        + b("in_emb", "Input Embedding", "× √d_model")
        + '<div style="text-align:center;opacity:0.7;font-size:0.85rem;">Inputs (English)</div>'
    )
    dec = (
        b("head", "Linear → Softmax", "over ~37k tokens")
        + '<div style="border:1px dashed #64748b;border-radius:12px;padding:8px;">'
        '<div style="font-size:0.75rem;opacity:0.7;text-align:right;">N = 6 ×</div>'
        + b("addnorm", "Add & Norm")
        + b("ffn", "Feed Forward", "512 → 2048 → 512", GREEN)
        + b("addnorm", "Add & Norm")
        + b("cross", "Multi-Head Attention", "Q: decoder · K,V: encoder", INDIGO)
        + b("addnorm", "Add & Norm")
        + b("dec_mha", "Masked Multi-Head Attention", "self, only earlier positions", BLUE)
        + "</div>"
        + b("pe", "⊕ Positional Encoding")
        + b("out_emb", "Output Embedding", "× √d_model")
        + '<div style="text-align:center;opacity:0.7;font-size:0.85rem;">Outputs, shifted right (German)</div>'
    )
    return (
        '<div style="display:flex;gap:18px;align-items:flex-end;">'
        f'<div style="flex:1;"><div style="text-align:center;font-weight:700;">Encoder</div>{enc}</div>'
        f'<div style="flex:1;"><div style="text-align:center;font-weight:700;">Decoder</div>{dec}</div>'
        "</div>"
    )


def _render_architecture() -> None:
    _owner("A")
    st.subheader("Figure 1: an encoder–decoder made only of attention and feed-forward layers")

    left, right = st.columns([1, 1])
    with right:
        sel_label = st.radio("Pick a component", list(COMPONENTS.values()), key="me_arch_sel")
        sel = [k for k, v in COMPONENTS.items() if v == sel_label][0]
        c1, c2, c3 = st.columns(3)
        B = c1.number_input("Batch B", 1, 64, 2, key="me_arch_b")
        n = c2.number_input("Source length n", 1, 200, 10, key="me_arch_n")
        m = c3.number_input("Target length m", 1, 200, 12, key="me_arch_m")
        d, dff, h, V = BASE["d_model"], BASE["d_ff"], BASE["h"], BASE["vocab"]

        info = {
            "in_emb": (
                "Each source token id becomes a learned vector of size d_model = 512. "
                "The same matrix is shared by the encoder embedding, the decoder embedding and the final linear layer (§3.4), "
                "and the embedding is multiplied by √d_model before positions are added.",
                r"x_i = \sqrt{d_{model}}\; E[\text{token}_i]",
                f"({B}, {n}) token ids → ({B}, {n}, {d})",
            ),
            "pe": (
                "Attention has no notion of order, so a fixed sinusoidal vector is **added** (not concatenated) to every embedding. "
                "Details in tab ③.",
                r"h_i^{(0)} = x_i + PE_i",
                f"({B}, {n}, {d}) + ({n}, {d}) → ({B}, {n}, {d})",
            ),
            "enc_mha": (
                "Queries, keys and values all come from the previous encoder layer, so every source word can look at every other source word. "
                "h = 8 heads of size 64.",
                r"\text{MultiHead}(X W^Q, X W^K, X W^V)",
                f"weights per head: ({B}, {h}, {n}, {n}) · output ({B}, {n}, {d})",
            ),
            "addnorm": (
                "Every sub-layer is wrapped in a residual connection followed by layer normalization (*post-LN*). "
                "Dropout is applied to the sub-layer output before the addition (§5.4). "
                "All sub-layers output size d_model = 512 so the residual sum is possible.",
                r"\text{LayerNorm}\big(x + \text{Dropout}(\text{Sublayer}(x))\big)",
                f"shape unchanged: (B, length, {d})",
            ),
            "ffn": (
                "The same two-layer network is applied to each position separately (equivalently: two 1×1 convolutions). "
                "Inner size d_ff = 2048. Different weights in every layer.",
                r"\text{FFN}(x) = \max(0,\; xW_1 + b_1)\,W_2 + b_2",
                f"({B}, {n}, {d}) → ({B}, {n}, {dff}) → ({B}, {n}, {d})",
            ),
            "out_emb": (
                "During training the decoder receives the correct translation **shifted right by one position** (a start token is prepended). "
                "Together with the mask, this means the prediction for position i can only use outputs before i (teacher forcing).",
                r"\text{input}_t = y_{t-1}, \qquad \text{target}_t = y_t",
                f"({B}, {m}) → ({B}, {m}, {d})",
            ),
            "dec_mha": (
                "Same as encoder self-attention, but scores for future positions are set to −∞ before the softmax, "
                "so their weight becomes exactly 0. This keeps the decoder auto-regressive while still training all positions in parallel.",
                r"\text{softmax}\!\left(\tfrac{QK^\top}{\sqrt{d_k}} + M\right)V,\quad M_{ij} = -\infty \text{ if } j > i",
                f"weights per head: ({B}, {h}, {m}, {m}), lower-triangular",
            ),
            "cross": (
                "Queries come from the decoder; keys and values come from the **final** encoder output. "
                "This is the classic Bahdanau-style attention between the two sentences, now computed in parallel.",
                r"\text{MultiHead}(Y W^Q,\; Z_{enc} W^K,\; Z_{enc} W^V)",
                f"weights per head: ({B}, {h}, {m}, {n}) — rectangular",
            ),
            "head": (
                "A linear layer (tied to the embedding matrix) maps each decoder position to a score for every vocabulary token; "
                "the softmax turns scores into next-token probabilities.",
                r"p(y_t \mid y_{<t}, x) = \text{softmax}(z_t E^\top)",
                f"({B}, {m}, {d}) → ({B}, {m}, {V:,})",
            ),
        }[sel]
        with st.container(border=True):
            st.markdown(f"**{sel_label}**")
            st.markdown(info[0])
            st.latex(info[1])
            st.code(info[2], language=None)

    with left:
        st.markdown(_arch_html(sel), unsafe_allow_html=True)

    st.markdown("#### Base model hyper-parameters")
    hp = pd.DataFrame(
        [
            ("N", 6, "layers in encoder and in decoder"),
            ("d_model", 512, "size of every token vector between sub-layers"),
            ("d_ff", 2048, "inner size of the feed-forward network"),
            ("h", 8, "attention heads"),
            ("d_k = d_v", 64, "size of each head = d_model / h"),
            ("P_drop", 0.1, "dropout rate"),
            ("ε_ls", 0.1, "label smoothing"),
        ],
        columns=["Symbol", "Base", "Meaning"],
    )
    st.dataframe(hp, use_container_width=True, hide_index=True)

    _speaker_notes(
        "- Read Figure 1 bottom-up: embed → add positions → N identical layers → output.\n"
        "- Click through the components on the right; the highlighted box follows. Change n and m to show that "
        "only cross-attention has a rectangular (m × n) weight matrix.\n"
        "- Three kinds of attention, one mechanism: they differ only in **where Q, K, V come from** and **the mask**.\n"
        "- Every sub-layer keeps size 512 so the residual addition works.\n"
        "- Bridge: *\"Let's look at the attention block itself.\"*"
    )


# ----------------------------------------------------------------------------------------
# ② Attention, three ways (Figure 2)
# ----------------------------------------------------------------------------------------
def _render_attention() -> None:
    _owner("A")
    st.subheader("Figure 2: scaled dot-product attention, then eight of them in parallel")

    c1, c2 = st.columns(2)
    with c1:
        with st.container(border=True):
            st.markdown("**Scaled dot-product attention** (Fig. 2, left)")
            st.latex(r"\text{Attention}(Q,K,V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V")
            st.markdown(
                "1. `MatMul` Q·Kᵀ: one similarity score per (query, key) pair  \n"
                "2. `Scale` by 1/√d_k  \n"
                "3. `Mask` (optional, decoder only)  \n"
                "4. `SoftMax` per row → weights summing to 1  \n"
                "5. `MatMul` with V → weighted average of values"
            )
            st.caption(
                "Why √d_k: if q and k have independent components with mean 0 and variance 1, q·k has variance d_k. "
                "Large scores push the softmax into regions with tiny gradients (footnote 4). "
                "Live demo: **Code → Scaled Dot-Product & Saturation**."
            )
    with c2:
        with st.container(border=True):
            st.markdown("**Multi-head attention** (Fig. 2, right)")
            st.latex(r"\text{head}_i = \text{Attention}(QW_i^Q,\; KW_i^K,\; VW_i^V)")
            st.latex(r"\text{MultiHead} = \text{Concat}(\text{head}_1,\dots,\text{head}_h)\,W^O")
            st.markdown(
                "Each head projects to a smaller space (d_k = d_v = d_model / h = 64), attends there, "
                "and the results are concatenated and mixed by W^O. "
                "*\"With a single attention head, averaging inhibits\"* attending to different things at once (§3.2.2)."
            )

    st.markdown("#### Splitting 512 dimensions into heads costs nothing extra")
    h = st.select_slider("Number of heads h", options=[1, 2, 4, 8, 16, 32], value=8, key="me_att_h")
    d = 512
    dk = d // h
    fig = go.Figure()
    for i in range(h):
        fig.add_trace(go.Bar(x=[dk], y=["d_model = 512"], orientation="h", marker_color=HEAD_COLORS[i % 8],
                             name=f"head {i + 1}", showlegend=False, hovertemplate=f"head {i + 1}: {dk} dims<extra></extra>"))
    fig.update_layout(barmode="stack", height=120, margin=dict(l=10, r=10, t=10, b=10), xaxis=dict(range=[0, 512]))
    st.plotly_chart(fig, use_container_width=True)
    m1, m2, m3 = st.columns(3)
    m1.metric("d_k = d_v per head", dk)
    m2.metric("Projection weights (Q,K,V,O)", f"{4 * d * d:,}", help="h · (3 · 512 · d_k) + (h · d_v) · 512 — independent of h")
    m3.metric("Score matrices per layer", f"{h} × (n × n)")
    st.caption("Total cost is (almost) the same for any h. What changes is how many different attention patterns a layer can express — tested in Table 3, rows (A).")

    st.markdown("#### The same mechanism, three ways (§3.2.3)")
    s1, s2 = st.columns(2)
    src = _tok(s1.text_input("Source sentence (encoder)", "The cat sat on the mat", key="me_att_src"), "The cat sat on the mat")
    tgt = _tok(s2.text_input("Target sentence (decoder)", "Die Katze saß auf der Matte", key="me_att_tgt"), "Die Katze saß auf der Matte")
    kind = st.radio(
        "Attention type",
        ["Encoder self-attention", "Masked decoder self-attention", "Encoder–decoder (cross) attention"],
        horizontal=True,
        key="me_att_kind",
    )
    if kind.startswith("Encoder self"):
        rows, cols = src, src
        allowed = np.ones((len(rows), len(cols)))
        note = "Q, K, V from the encoder. Every source word sees every source word: square n × n, no mask."
    elif kind.startswith("Masked"):
        rows, cols = tgt, tgt
        allowed = np.tril(np.ones((len(rows), len(cols))))
        note = "Q, K, V from the decoder. Word i sees only words ≤ i (upper triangle set to −∞ → weight 0): square m × m."
    else:
        rows, cols = tgt, src
        allowed = np.ones((len(rows), len(cols)))
        note = "Q from the decoder, K and V from the final encoder output. Each target word can look at the whole source: rectangular m × n."

    fig = go.Figure(go.Heatmap(
        z=allowed, x=[f"{j + 1}. {t}" for j, t in enumerate(cols)], y=[f"{i + 1}. {t}" for i, t in enumerate(rows)],
        colorscale=[[0, "#1f2937"], [1, BLUE]], zmin=0, zmax=1, showscale=False, xgap=2, ygap=2,
        hovertemplate="query %{y} → key %{x}<extra></extra>",
    ))
    fig.update_layout(height=120 + 32 * len(rows), margin=dict(l=10, r=10, t=30, b=10),
                      yaxis=dict(autorange="reversed", title="queries"), xaxis=dict(title="keys", side="top"),
                      title="Which (query, key) pairs are allowed (blue)")
    st.plotly_chart(fig, use_container_width=True)
    st.info(note)

    _speaker_notes(
        "- Walk Figure 2 left box by box: MatMul, Scale, Mask, SoftMax, MatMul.\n"
        "- Move the heads slider: the 512 dimensions are cut into slices, the parameter count does not change. "
        "This is exactly the 'constant computation' condition of Table 3 (A).\n"
        "- Switch between the three attention types: same formula, different Q/K/V sources and masks. Point out the rectangular cross-attention matrix.\n"
        "- Don't re-derive √d_k here; the Code section has the live demo.\n"
        "- Bridge: *\"None of this knows word order. That's the next piece.\"*"
    )


# ----------------------------------------------------------------------------------------
# ③ Word order, embeddings, FFN
# ----------------------------------------------------------------------------------------
def _positional_encoding(n_pos: int, d: int) -> np.ndarray:
    pos = np.arange(n_pos)[:, None]
    i = np.arange(0, d, 2)[None, :]
    angle = pos / np.power(10000.0, i / d)
    pe = np.zeros((n_pos, d))
    pe[:, 0::2] = np.sin(angle)
    pe[:, 1::2] = np.cos(angle)
    return pe


def _render_positions() -> None:
    _owner("A")
    st.subheader("Positional encoding: giving word order back to a model that has none")
    st.markdown("Self-attention treats its input as a **set**: shuffle the words and the outputs shuffle the same way. "
                "So the authors add a position signal to each embedding (§3.5):")
    st.latex(r"PE_{(pos,\,2i)} = \sin\!\left(\frac{pos}{10000^{2i/d_{model}}}\right),\qquad PE_{(pos,\,2i+1)} = \cos\!\left(\frac{pos}{10000^{2i/d_{model}}}\right)")

    c1, c2 = st.columns(2)
    n_pos = c1.slider("Positions shown", 10, 200, 100, key="me_pe_n")
    d = c2.select_slider("d_model", options=[64, 128, 256, 512], value=512, key="me_pe_d")
    pe = _positional_encoding(n_pos, d)

    fig = go.Figure(go.Heatmap(z=pe, colorscale="RdBu", zmid=0, colorbar=dict(title="value"),
                               hovertemplate="pos %{y}, dim %{x}: %{z:.2f}<extra></extra>"))
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=30, b=10), xaxis_title="embedding dimension",
                      yaxis_title="position", yaxis=dict(autorange="reversed"), title="Every row is the vector added to the word at that position")
    st.plotly_chart(fig, use_container_width=True)

    l, r = st.columns(2)
    with l:
        dims = st.multiselect("Dimensions to plot", options=list(range(0, 512, 2)), default=[0, 16, 62, 254], key="me_pe_dims")
        dims = [x for x in dims if x < d]
        fig = go.Figure()
        for k, dim in enumerate(dims):
            fig.add_trace(go.Scatter(x=list(range(n_pos)), y=pe[:, dim], mode="lines", name=f"dim {dim}",
                                     line=dict(color=HEAD_COLORS[k % 8])))
        fig.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=10), xaxis_title="position",
                          title="Each dimension is a wave; low dims fast, high dims slow")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Wavelengths form a geometric progression from 2π to 10000·2π.")
    with r:
        fig = go.Figure()
        ks = np.arange(0, min(60, n_pos - 1))
        for k_i, p in enumerate([0, 20, 40]):
            if p + ks[-1] < n_pos:
                fig.add_trace(go.Scatter(x=ks, y=[pe[p] @ pe[p + k] for k in ks], mode="lines", name=f"from pos {p}",
                                         line=dict(color=HEAD_COLORS[k_i], width=6 - 2 * k_i)))
        fig.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=10), xaxis_title="offset k",
                          yaxis_title="PE(pos) · PE(pos + k)", title="Similarity depends only on the distance k")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("The three curves lie exactly on top of each other: since sin·sin + cos·cos = cos(difference), "
                   "the dot product depends only on k. The paper's motivation: PE(pos + k) is a linear function of PE(pos).")

    st.markdown("#### Why sinusoids and not learned positions?")
    st.markdown(
        "The authors tried **learned positional embeddings** too: *\"nearly identical results\"* (Table 3, row E: 25.7 vs 25.8 BLEU). "
        "They kept sinusoids because they *may* let the model handle sentences longer than any seen in training."
    )

    st.markdown("#### The two other per-token pieces")
    c1, c2 = st.columns(2)
    with c1:
        with st.container(border=True):
            st.markdown("**Shared embeddings (§3.4)**")
            st.markdown("One matrix (vocab × 512) is used three times: encoder input, decoder input, and the pre-softmax output layer "
                        "(weight tying, Press & Wolf 2017). Possible because English–German uses a **shared** BPE vocabulary. "
                        "Embeddings are scaled by √512 ≈ 22.6 so they are not drowned by the positional signal.")
    with c2:
        with st.container(border=True):
            st.markdown("**Position-wise FFN (§3.3)**")
            st.markdown("Attention mixes information **between** positions; the FFN transforms each position **on its own**. "
                        "With d_ff = 2048 it holds about two thirds of each encoder layer's weights.")

    _speaker_notes(
        "- Start with the problem: without positions, 'dog bites man' = 'man bites dog' for attention.\n"
        "- Heatmap: each row is one position's fingerprint. Left columns change fast, right columns slowly — like the hands of a clock.\n"
        "- Right plot: the three curves overlap perfectly. That is the mathematical reason sinusoids encode **relative** position.\n"
        "- Row E of Table 3: learned positions work just as well; the choice was about extrapolation, not accuracy.\n"
        "- Bridge: *\"With the model built and trained, the authors looked inside it.\"*"
    )


# ----------------------------------------------------------------------------------------
# ④ Reading the attention figures (Figures 3-5)
# ----------------------------------------------------------------------------------------
FIG_SENTENCE = "The Law will never be perfect , but its application should be just".split()


def _toy_heads(tokens: list[str]) -> dict[str, np.ndarray]:
    """Hand-made illustrative patterns. NOT weights from the trained model."""
    n = len(tokens)
    prev = np.zeros((n, n))
    for i in range(n):
        prev[i, max(i - 1, 0)] = 1.0
    nxt = np.zeros((n, n))
    for i in range(n):
        nxt[i, min(i + 1, n - 1)] = 1.0
    coref = np.full((n, n), 0.02)
    lower = [t.lower() for t in tokens]
    for i in range(n):
        coref[i, i] += 0.5
    if "its" in lower and "law" in lower:
        coref[lower.index("its"), :] = 0.01
        coref[lower.index("its"), lower.index("law")] = 0.9
    coref = coref / coref.sum(axis=1, keepdims=True)
    return {"previous word": prev, "next word": nxt, "'its' → 'Law' (coreference-like)": coref}


def _bipartite(tokens: list[str], heads: dict[str, np.ndarray], query: int | None) -> go.Figure:
    n = len(tokens)
    fig = go.Figure()
    for k, (name, w) in enumerate(heads.items()):
        color = HEAD_COLORS[k % 8]
        first = True
        for i in range(n):
            if query is not None and i != query:
                continue
            for j in range(n):
                if w[i, j] < 0.05:
                    continue
                fig.add_trace(go.Scatter(x=[0, 1], y=[i, j], mode="lines", line=dict(color=color, width=1 + 6 * w[i, j]),
                                         opacity=float(min(1, 0.15 + w[i, j])), name=name, legendgroup=name, showlegend=first,
                                         hovertemplate=f"{name}: {tokens[i]} → {tokens[j]} = {w[i, j]:.2f}<extra></extra>"))
                first = False
    for side in (0, 1):
        fig.add_trace(go.Scatter(x=[side] * n, y=list(range(n)), mode="text", text=tokens,
                                 textposition="middle left" if side == 0 else "middle right", showlegend=False, hoverinfo="skip"))
    fig.update_layout(height=40 + 30 * n, margin=dict(l=10, r=10, t=30, b=10),
                      xaxis=dict(visible=False, range=[-0.6, 1.6]), yaxis=dict(visible=False, autorange="reversed"),
                      legend=LEGEND)
    return fig


def _render_figures() -> None:
    _owner("A")
    st.subheader("Figures 3–5: how the attention visualisations were made")
    st.markdown(
        "The appendix shows attention weights of the **trained English→German model**. All three figures use the "
        "**encoder self-attention of layer 5 (of 6)** on a single input sentence. Each figure is a qualitative example, not a measurement."
    )
    figs = pd.DataFrame(
        [
            ("Figure 3", "Long-distance dependency", "Only the word 'making'; colours = different heads", "Many heads link 'making' to the distant 'more difficult'"),
            ("Figure 4", "Anaphora (coreference)", "Top: all words, head 5. Bottom: only the word 'its', heads 5 and 6", "'its' attends sharply to 'Law' / 'application'"),
            ("Figure 5", "Sentence structure", "All words, two different heads", "Heads learn different, structure-like patterns"),
        ],
        columns=["Figure", "Phenomenon", "What is drawn", "Claim in the caption"],
    )
    st.dataframe(figs, use_container_width=True, hide_index=True)

    st.markdown("#### How to read one: the drawing recipe")
    st.markdown(
        "1. Run the sentence through the encoder and keep the softmax matrix **A** of one layer (shape heads × n × n).  \n"
        "2. Write the tokens twice: left column = queries, right column = keys.  \n"
        "3. For each head, draw a line from word i to word j with **thickness / opacity ∝ A[i, j]**; one colour per head.  \n"
        "4. Optionally keep only the lines of one query word (Fig. 3, Fig. 4 bottom)."
    )

    st.warning("The weights below are **hand-made illustrative patterns** to show how the drawing works. "
               "They are not from the trained Transformer. Real weights from a trained model: **Code → BertViz Multiscale Explorer**.")
    tokens = FIG_SENTENCE
    heads = _toy_heads(tokens)
    c1, c2 = st.columns([1, 2])
    with c1:
        chosen = st.multiselect("Heads", list(heads.keys()), default=list(heads.keys()), key="me_fig_heads")
        only = st.toggle("Show one query word only (like Fig. 3)", value=True, key="me_fig_only")
        q = st.selectbox("Query word", list(range(len(tokens))), index=tokens.index("its"),
                         format_func=lambda i: tokens[i], key="me_fig_q", disabled=not only)
    with c2:
        st.plotly_chart(_bipartite(tokens, {k: heads[k] for k in chosen}, q if only else None), use_container_width=True)

    st.markdown("#### Extracting real weights yourself")
    st.code(
        """from transformers import AutoTokenizer, AutoModel
tok = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased", output_attentions=True)
out = model(**tok("The Law will never be perfect, but its application should be just", return_tensors="pt"))
A = out.attentions[4]        # layer 5 -> tensor (batch, heads, n, n), each row sums to 1
""",
        language="python",
    )

    st.markdown("#### Methodological caveats")
    st.markdown(
        "- **Cherry-picked**: a few heads, one layer, one sentence each; no statistics over a dataset.\n"
        "- **Attention ≠ explanation**: different attention maps can give the same prediction (Jain & Wallace 2019), "
        "though the question is debated (Wiegreffe & Pinter 2019).\n"
        "- Weights show where information is **read from**, not how the value vectors are then transformed."
    )

    _speaker_notes(
        "- Same model as Table 2, encoder, layer 5 of 6. Say this explicitly: it's the most common question.\n"
        "- Explain the drawing recipe with the toy plot: thick line = high weight. Toggle 'one query word' to reproduce the style of Fig. 3 / Fig. 4 bottom.\n"
        "- Stress that our toy weights are illustrative; for real weights point to the BertViz tab.\n"
        "- Close Part A with the caveats: these figures are suggestive, not evidence of 'understanding'.\n"
        f"- Hand over: *\"{PRESENTERS['B']} will now explain how the model was trained and evaluated.\"*"
    )


# ----------------------------------------------------------------------------------------
# ⑤ Data & batching
# ----------------------------------------------------------------------------------------
DATASETS = [
    ("WMT 2014 English→German", "≈ 4.5M sentence pairs", "Byte-pair encoding (BPE)", "≈ 37,000 tokens, shared source + target", "newstest2013 (dev) / newstest2014 (test)"),
    ("WMT 2014 English→French", "36M sentences", "Word-pieces", "32,000 tokens", "newstest2014 (test)"),
    ("WSJ Penn Treebank (parsing)", "≈ 40k sentences", "—", "16,000 tokens", "Section 22 (dev) / Section 23 (test)"),
    ("WSJ + high-confidence & BerkeleyParser corpora", "≈ 17M sentences", "—", "32,000 tokens", "Section 22 (dev) / Section 23 (test)"),
]

BPE_CORPUS = {"l o w </w>": 5, "l o w e r </w>": 2, "n e w e s t </w>": 6, "w i d e s t </w>": 3}


def _bpe_merges(corpus: dict[str, int], n_merges: int) -> tuple[list[tuple[str, str, int]], dict[str, int]]:
    vocab = dict(corpus)
    merges = []
    for _ in range(n_merges):
        pairs = Counter()
        for word, freq in vocab.items():
            sym = word.split()
            for a, b in zip(sym, sym[1:]):
                pairs[(a, b)] += freq
        if not pairs:
            break
        (a, b), f = max(pairs.items(), key=lambda kv: (kv[1], kv[0]))
        merges.append((a, b, f))
        new_vocab = {}
        for word, freq in vocab.items():
            sym, out, i = word.split(), [], 0
            while i < len(sym):
                if i < len(sym) - 1 and sym[i] == a and sym[i + 1] == b:
                    out.append(a + b)
                    i += 2
                else:
                    out.append(sym[i])
                    i += 1
            new_vocab[" ".join(out)] = freq
        vocab = new_vocab
    return merges, vocab


def _bpe_apply(word: str, merges: list[tuple[str, str, int]]) -> list[str]:
    sym = list(word) + ["</w>"]
    for a, b, _ in merges:
        out, i = [], 0
        while i < len(sym):
            if i < len(sym) - 1 and sym[i] == a and sym[i + 1] == b:
                out.append(a + b)
                i += 2
            else:
                out.append(sym[i])
                i += 1
        sym = out
    return sym


def _batches(lengths: np.ndarray, budget: int) -> list[np.ndarray]:
    batches, cur, cur_max = [], [], 0
    for L in lengths:
        new_max = max(cur_max, L)
        if cur and new_max * (len(cur) + 1) > budget:
            batches.append(np.array(cur))
            cur, new_max = [], L
        cur.append(L)
        cur_max = new_max
    if cur:
        batches.append(np.array(cur))
    return batches


def _padding_ratio(batches: list[np.ndarray]) -> float:
    real = sum(b.sum() for b in batches)
    padded = sum(len(b) * b.max() for b in batches)
    return 1 - real / padded


def _render_data() -> None:
    _owner("B")
    st.subheader("Training data and batching (§5.1)")
    st.dataframe(pd.DataFrame(DATASETS, columns=["Dataset", "Size", "Tokenisation", "Vocabulary", "Dev / test"]),
                 use_container_width=True, hide_index=True)
    st.caption("WMT = Workshop on Machine Translation shared task; newstestYYYY = news articles translated by professionals, used only for evaluation.")

    st.markdown("#### Byte-pair encoding: how a 37k vocabulary is built")
    st.markdown("Start from characters, then repeatedly **merge the most frequent adjacent pair**. "
                "Frequent words end up as one token, rare words are split into known pieces, so there are no unknown words.")
    n_merges = st.slider("Number of merges", 0, 12, 4, key="me_bpe_n")
    merges, vocab = _bpe_merges(BPE_CORPUS, n_merges)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Toy corpus after merging** (word · count)")
        st.dataframe(pd.DataFrame([(w, f) for w, f in vocab.items()], columns=["Segmentation", "Count"]),
                     use_container_width=True, hide_index=True)
        st.markdown("**Merges learned, in order**")
        st.code("\n".join(f"{i + 1:>2}. {a} + {b}  (seen {f}×)" for i, (a, b, f) in enumerate(merges)) or "(none yet)", language=None)
    with c2:
        st.markdown("**Applying the merges to new words**")
        for w in ["lowest", "newer", "wider"]:
            st.markdown(f"`{w}` → " + " · ".join(f"`{s}`" for s in _bpe_apply(w, merges)))
        st.caption("Toy corpus from Sennrich et al. (2016). The real EN→DE vocabulary used ~37k merges on millions of sentences.")

    st.markdown("#### Batching by length: why ~25,000 tokens and not 'N sentences'")
    st.markdown("A batch is a rectangle (sentences × longest sentence). Shorter sentences get **padding** — wasted compute. "
                "The paper groups sentences of similar length so each batch holds ≈ 25,000 source + 25,000 target tokens.")
    c1, c2 = st.columns(2)
    budget = c1.select_slider("Token budget per batch", options=[2000, 5000, 10000, 25000], value=25000, key="me_batch_budget")
    seed = c2.number_input("Random seed", 0, 999, 0, key="me_batch_seed")
    rng = np.random.default_rng(int(seed))
    lengths = np.clip(rng.lognormal(mean=3.1, sigma=0.55, size=20000).astype(int), 2, 200)
    rand_b = _batches(rng.permutation(lengths), budget)
    sort_b = _batches(np.sort(lengths), budget)
    m1, m2, m3 = st.columns(3)
    m1.metric("Padding, random batches", f"{_padding_ratio(rand_b):.0%}")
    m2.metric("Padding, length-sorted batches", f"{_padding_ratio(sort_b):.0%}")
    m3.metric("Sentences per batch (sorted)", f"{min(len(b) for b in sort_b)} – {max(len(b) for b in sort_b)}")

    def batch_img(b: np.ndarray, k: int = 40) -> np.ndarray:
        b = b[:k]
        img = np.zeros((len(b), b.max()))
        for i, L in enumerate(b):
            img[i, :L] = 1
        return img

    l, r = st.columns(2)
    for col, title, b in ((l, "One random batch", rand_b[len(rand_b) // 2]), (r, "One length-sorted batch", sort_b[len(sort_b) // 2])):
        with col:
            fig = go.Figure(go.Heatmap(z=batch_img(b), colorscale=[[0, "#1f2937"], [1, GREEN]], showscale=False, zmin=0, zmax=1))
            fig.update_layout(height=260, margin=dict(l=10, r=10, t=30, b=10), title=f"{title} (green = real token, dark = padding)",
                              xaxis_title="position", yaxis_title="sentence (first 40)")
            st.plotly_chart(fig, use_container_width=True)
    st.caption("Simulation: 20,000 synthetic sentence lengths (log-normal, median ≈ 22 tokens), not WMT data. The paper does not report its padding rate.")

    _speaker_notes(
        "- Two translation datasets, very different sizes: 4.5M vs 36M pairs. EN→DE is the main benchmark; all ablations use it.\n"
        "- BPE demo: move the slider; frequent pieces like 'est</w>' get merged first. Show that 'lowest' becomes known pieces even though it's not in the corpus.\n"
        "- Shared EN–DE vocabulary is what allows the three-way weight tying from tab ③.\n"
        "- Batching demo: random batching wastes a large share of compute on padding; sorting by length almost eliminates it. "
        "Batch size is measured in **tokens**, so the number of sentences per batch varies.\n"
        "- Bridge: *\"Next: how they optimised on these batches.\"*"
    )


# ----------------------------------------------------------------------------------------
# ⑥ Optimizer, schedule, hardware
# ----------------------------------------------------------------------------------------
def _lr(step: np.ndarray, d: int, warmup: int) -> np.ndarray:
    step = np.maximum(step, 1)
    return d ** -0.5 * np.minimum(step ** -0.5, step * warmup ** -1.5)


def _render_training() -> None:
    _owner("B")
    st.subheader("Optimizer and learning-rate schedule (§5.3)")
    st.markdown("Adam with **β₁ = 0.9, β₂ = 0.98, ε = 10⁻⁹**, and a learning rate that is never set by hand:")
    st.latex(r"lrate = d_{model}^{-0.5} \cdot \min\!\left(step^{-0.5},\; step \cdot warmup\_steps^{-1.5}\right)")
    st.markdown("→ **linear warm-up** for the first `warmup_steps`, then **decay ∝ 1/√step**.")

    c1, c2, c3 = st.columns(3)
    d = c1.select_slider("d_model", options=[256, 512, 1024], value=512, key="me_lr_d")
    warm = c2.select_slider("warmup_steps", options=[500, 1000, 2000, 4000, 8000, 16000], value=4000, key="me_lr_w")
    total = c3.select_slider("Training steps", options=[100_000, 300_000], value=100_000,
                             format_func=lambda s: f"{s // 1000}k ({'base' if s == 100_000 else 'big'})", key="me_lr_t")
    steps = np.linspace(1, total, 1500)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=steps, y=_lr(steps, 512, 4000), name="paper (d=512, warm-up 4000)", line=dict(color=SLATE, dash="dash")))
    fig.add_trace(go.Scatter(x=steps, y=_lr(steps, d, warm), name="your setting", line=dict(color=AMBER, width=3)))
    peak = d ** -0.5 * warm ** -0.5
    fig.add_trace(go.Scatter(x=[warm], y=[peak], mode="markers+text", text=[f"peak {peak:.2e}"], textposition="top right",
                             marker=dict(color=AMBER, size=10), showlegend=False))
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=30, b=10), xaxis_title="training step", yaxis_title="learning rate",
                      legend=LEGEND)
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"Peak = (d_model · warmup_steps)^−0.5. For the paper's base model: (512 · 4000)^−0.5 ≈ 7.0 × 10⁻⁴, reached at step 4,000.")

    c1, c2 = st.columns(2)
    with c1:
        with st.container(border=True):
            st.markdown("**Why warm up?**")
            st.markdown("Early in training Adam's variance estimates are based on few gradients and the post-LN Transformer has large gradients near the output. "
                        "A full learning rate at step 1 tends to diverge. Later work tied the need for warm-up to post-LN (Xiong et al., 2020).")
    with c2:
        with st.container(border=True):
            st.markdown("**Why scale by d_model^−0.5?**")
            st.markdown("Wider models get a smaller learning rate automatically, so one formula serves base (512) and big (1024).")

    st.markdown("#### Hardware and schedule (§5.2)")
    st.markdown("One machine with **8 NVIDIA P100 GPUs**.")
    c1, c2, c3 = st.columns(3)
    preset = c1.radio("Model", ["Base", "Big"], horizontal=True, key="me_hw_preset")
    step_t = c2.number_input("Seconds per step", 0.1, 5.0, 0.4 if preset == "Base" else 1.0, 0.1, key=f"me_hw_t_{preset}")
    n_steps = c3.number_input("Steps", 10_000, 1_000_000, 100_000 if preset == "Base" else 300_000, 10_000, key=f"me_hw_s_{preset}")
    hours = step_t * n_steps / 3600
    m1, m2, m3 = st.columns(3)
    m1.metric("Wall-clock time", f"{hours:.1f} h" if hours < 48 else f"{hours / 24:.1f} days")
    m2.metric("Tokens seen (≈ 25k src + 25k tgt per step)", f"{n_steps * 50_000 / 1e9:.1f} B")
    m3.metric("GPU-hours", f"{hours * 8:,.0f}")
    st.caption("Paper: base 0.4 s/step × 100k steps ≈ 12 h; big 1.0 s/step × 300k steps = 3.5 days.")

    _speaker_notes(
        "- One formula, two phases. Move warm-up to 500: the peak gets much higher and earlier — that's the setting that tends to diverge.\n"
        "- The peak value is fixed by d_model and warm-up; nobody tunes a 'learning rate' directly.\n"
        "- Hardware: 8 GPUs of 2016 vintage, 12 hours for the base model. The cheap training is part of the paper's claim.\n"
        "- Bridge: *\"With 65M parameters and 4.5M sentence pairs, overfitting is the next worry.\"*"
    )


# ----------------------------------------------------------------------------------------
# ⑦ Regularization
# ----------------------------------------------------------------------------------------
def _render_regularization() -> None:
    _owner("B")
    st.subheader("Regularization (§5.4): residual dropout and label smoothing")

    c1, c2 = st.columns(2)
    with c1:
        with st.container(border=True):
            st.markdown("**Residual dropout, P_drop = 0.1**")
            st.markdown("Applied in two places:\n"
                        "1. to the **output of every sub-layer**, before it is added to the residual and normalised;\n"
                        "2. to the **sum of embeddings + positional encodings**, in encoder and decoder.")
            st.latex(r"\text{LayerNorm}\big(x + \text{Dropout}_{0.1}(\text{Sublayer}(x))\big)")
            st.caption("Big EN→DE model: P_drop = 0.3. Big EN→FR: 0.1 (§6.1).")
    with c2:
        with st.container(border=True):
            st.markdown("**Label smoothing, ε_ls = 0.1** (Szegedy et al., 2016)")
            st.markdown("The training target is no longer 100% on the correct token; ε is spread uniformly over the vocabulary of size K:")
            st.latex(r"q'(k) = (1-\varepsilon)\,\mathbb{1}[k = y] + \frac{\varepsilon}{K}")
            st.markdown("*\"This hurts perplexity, as the model learns to be more unsure, but improves accuracy and BLEU score.\"*")

    st.markdown("#### Why label smoothing hurts perplexity but helps BLEU")
    c1, c2 = st.columns([1, 2])
    with c1:
        eps = st.slider("ε_ls", 0.0, 0.3, 0.1, 0.05, key="me_ls_eps")
        K = st.select_slider("Vocabulary size K", options=[10, 100, 1000, 37000], value=37000, key="me_ls_k")
        p_opt = 1 - eps + eps / K
        st.metric("Best achievable p(correct token)", f"{p_opt:.3f}")
        st.metric("→ perplexity floor vs true labels", f"{1 / p_opt:.3f}", help="exp(−log p_opt): what an ideal smoothed model scores on one-hot targets")
    with c2:
        p = np.linspace(0.01, 0.999, 400)
        rest = (1 - p) / (K - 1)
        ce_smooth = -((1 - eps + eps / K) * np.log(p) + (eps - eps / K) * np.log(rest))
        ce_hard = -np.log(p)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=p, y=ce_hard, name="loss vs one-hot target (ε = 0)", line=dict(color=SLATE, dash="dash")))
        fig.add_trace(go.Scatter(x=p, y=ce_smooth, name=f"loss vs smoothed target (ε = {eps})", line=dict(color=AMBER, width=3)))
        fig.add_vline(x=p_opt, line=dict(color=AMBER, dash="dot"))
        fig.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10), xaxis_title="probability the model gives the correct token",
                          yaxis_title="cross-entropy", yaxis=dict(range=[0, 8]), legend=LEGEND,
                          title="With smoothing, the loss is minimised before p = 1")
        st.plotly_chart(fig, use_container_width=True)
    st.caption("Remaining probability spread evenly over the other K−1 tokens. With smoothing the model is penalised for being 100% sure, "
               "so perplexity (measured against the real one-hot labels) gets worse, while calibrated, less over-confident scores help beam search.")
    st.info("Evidence in the paper — Table 3, row (D): ε_ls = 0 gives **better** perplexity (4.67 vs 4.92) but **worse** BLEU (25.3 vs 25.8). "
            "Dropout 0.0 → BLEU 24.6: dropout matters most.")

    _speaker_notes(
        "- Dropout: placed right before each residual addition, and on embeddings + positions. 0.1 for base, 0.3 for big EN→DE.\n"
        "- Label smoothing: move ε. The orange curve has its minimum before p = 1, so the model is trained never to be fully sure.\n"
        "- That is exactly why perplexity gets worse and BLEU better — and Table 3 row (D) confirms both directions.\n"
        "- Bridge: *\"Training gives us a set of weights. How do we turn them into a translation and a score?\"*"
    )


# ----------------------------------------------------------------------------------------
# ⑧ Decoding & evaluation protocol
# ----------------------------------------------------------------------------------------
GPUS = {"K80": 2.8, "K40": 3.7, "M40": 6.0, "P100": 9.5}


def _render_decoding() -> None:
    _owner("B")
    st.subheader("From trained weights to a BLEU score (§6.1)")
    st.markdown(_chips(["Checkpoint averaging", "Beam search (k = 4, α = 0.6)", "Max length = input + 50", "BLEU on newstest2014"], GREEN),
                unsafe_allow_html=True)

    st.markdown("#### 1 · Checkpoint averaging")
    st.markdown("Instead of the final weights, the reported model is the **average of the last checkpoints**, saved every 10 minutes: "
                "**5** for base models, **20** for big models. Averaging smooths out the noise of the last optimisation steps.")
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Base: steps between checkpoints", f"{600 / 0.4:,.0f}")
        st.metric("Base: averaged window", f"last ≈ {5 * 600 / 0.4:,.0f} steps (≈ 50 min)")
    with c2:
        st.metric("Big: steps between checkpoints", f"{600 / 1.0:,.0f}")
        st.metric("Big: averaged window", f"last ≈ {20 * 600 / 1.0:,.0f} steps (≈ 3.3 h)")
    st.caption("Derived from 10-minute intervals and the step times in §5.2. Table 3 (ablations) uses **no** checkpoint averaging.")

    st.markdown("#### 2 · Beam search with a length penalty")
    st.markdown("The decoder keeps the **4 best partial translations** at each step. Because every extra token multiplies in a probability < 1, "
                "raw scores favour short outputs. GNMT's length penalty (Wu et al., 2016) divides by a growing factor:")
    st.latex(r"\text{score}(Y) = \frac{\log P(Y \mid X)}{lp(Y)}, \qquad lp(Y) = \frac{(5 + |Y|)^{\alpha}}{(5+1)^{\alpha}}")
    c1, c2 = st.columns([1, 2])
    with c1:
        alpha = st.slider("α", 0.0, 1.5, 0.6, 0.1, key="me_bs_a")
        ls, ps = st.number_input("Short hypothesis: length", 1, 50, 6, key="me_bs_ls"), st.number_input("… log P", -50.0, 0.0, -4.0, 0.5, key="me_bs_ps")
        ll, pl = st.number_input("Long hypothesis: length", 1, 50, 12, key="me_bs_ll"), st.number_input("… log P", -50.0, 0.0, -5.0, 0.5, key="me_bs_pl")
    lp = lambda L: ((5 + L) / 6) ** alpha
    s_short, s_long = ps / lp(ls), pl / lp(ll)
    with c2:
        fig = go.Figure(go.Bar(x=["short", "long"], y=[s_short, s_long], marker_color=[BLUE if s_short > s_long else SLATE, BLUE if s_long > s_short else SLATE],
                               text=[f"{s_short:.2f}", f"{s_long:.2f}"], textposition="outside", cliponaxis=False))
        fig.update_layout(height=260, margin=dict(l=10, r=10, t=30, b=10), yaxis_title="normalised score (higher wins)",
                          title=f"Winner at α = {alpha}: {'short' if s_short > s_long else 'long'} hypothesis")
        st.plotly_chart(fig, use_container_width=True)
    st.caption("At α = 0 the short hypothesis wins; at the paper's α = 0.6 the longer, more complete one wins. "
               "k = 4 and α = 0.6 were chosen on the development set. Output length is capped at input length + 50.")

    st.markdown("#### 3 · BLEU")
    st.markdown("BLEU compares the output with a professional reference translation: the geometric mean of 1- to 4-gram precisions, "
                "times a brevity penalty for outputs shorter than the reference. Scale 0–100; on newstest2014 EN→DE, +1 BLEU was considered a large gain.")

    st.markdown("#### 4 · The 'Training cost' column of Table 2")
    st.markdown("Not measured — **estimated** as *training time × number of GPUs × sustained single-precision TFLOPS per GPU* (footnote 5).")
    c1, c2, c3, c4 = st.columns(4)
    preset = c1.radio("Preset", ["Base", "Big"], horizontal=True, key="me_cost_p")
    gpu = c2.selectbox("GPU", list(GPUS.keys()), index=3, key="me_cost_gpu")
    n_gpu = c3.number_input("GPUs", 1, 128, 8, key="me_cost_n")
    hrs = c4.number_input("Hours", 1.0, 2000.0, 12.0 if preset == "Base" else 84.0, key=f"me_cost_h_{preset}")
    flops = hrs * 3600 * n_gpu * GPUS[gpu] * 1e12
    st.metric("Estimated training FLOPs", f"{flops:.2e}")
    st.caption("Paper, Table 2: base 3.3 · 10¹⁸, big 2.3 · 10¹⁹. Assumed TFLOPS: K80 2.8, K40 3.7, M40 6.0, P100 9.5. "
               "Costs of competing models were computed the same way from their papers' reported times and GPUs.")

    _speaker_notes(
        "- Four steps between training and the number in Table 2. Each one is a methodological choice.\n"
        "- Checkpoint averaging: free ensemble-like smoothing. Important: Table 3 does **not** use it, so Table 3 numbers are lower than Table 2.\n"
        "- Length penalty demo: at α = 0 the short hypothesis wins; at 0.6 the long one wins.\n"
        "- Training cost: show that 12 h × 8 × 9.5 TFLOPS gives 3.3e18 — the column is an upper-bound estimate, not a measurement.\n"
        "- Results themselves: Roman's tab.\n"
        "- Bridge: *\"How do we know which design choices matter? Table 3.\"*"
    )


# ----------------------------------------------------------------------------------------
# ⑨ Model variations (Table 3)
# ----------------------------------------------------------------------------------------
TABLE_3 = [
    # group, label, N, d_model, d_ff, h, d_k, d_v, P_drop, eps_ls, steps, PPL, BLEU, params(M)
    ("base", "base", 6, 512, 2048, 8, 64, 64, 0.1, 0.1, "100K", 4.92, 25.8, 65),
    ("A", "h=1, d_k=d_v=512", 6, 512, 2048, 1, 512, 512, 0.1, 0.1, "100K", 5.29, 24.9, None),
    ("A", "h=4, d_k=d_v=128", 6, 512, 2048, 4, 128, 128, 0.1, 0.1, "100K", 5.00, 25.5, None),
    ("A", "h=16, d_k=d_v=32", 6, 512, 2048, 16, 32, 32, 0.1, 0.1, "100K", 4.91, 25.8, None),
    ("A", "h=32, d_k=d_v=16", 6, 512, 2048, 32, 16, 16, 0.1, 0.1, "100K", 5.01, 25.4, None),
    ("B", "d_k=16", 6, 512, 2048, 8, 16, 64, 0.1, 0.1, "100K", 5.16, 25.1, 58),
    ("B", "d_k=32", 6, 512, 2048, 8, 32, 64, 0.1, 0.1, "100K", 5.01, 25.4, 60),
    ("C", "N=2", 2, 512, 2048, 8, 64, 64, 0.1, 0.1, "100K", 6.11, 23.7, 36),
    ("C", "N=4", 4, 512, 2048, 8, 64, 64, 0.1, 0.1, "100K", 5.19, 25.3, 50),
    ("C", "N=8", 8, 512, 2048, 8, 64, 64, 0.1, 0.1, "100K", 4.88, 25.5, 80),
    ("C", "d_model=256, d_k=d_v=32", 6, 256, 2048, 8, 32, 32, 0.1, 0.1, "100K", 5.75, 24.5, 28),
    ("C", "d_model=1024, d_k=d_v=128", 6, 1024, 2048, 8, 128, 128, 0.1, 0.1, "100K", 4.66, 26.0, 168),
    ("C", "d_ff=1024", 6, 512, 1024, 8, 64, 64, 0.1, 0.1, "100K", 5.12, 25.4, 53),
    ("C", "d_ff=4096", 6, 512, 4096, 8, 64, 64, 0.1, 0.1, "100K", 4.75, 26.2, 90),
    ("D", "P_drop=0.0", 6, 512, 2048, 8, 64, 64, 0.0, 0.1, "100K", 5.77, 24.6, None),
    ("D", "P_drop=0.2", 6, 512, 2048, 8, 64, 64, 0.2, 0.1, "100K", 4.95, 25.5, None),
    ("D", "ε_ls=0.0", 6, 512, 2048, 8, 64, 64, 0.1, 0.0, "100K", 4.67, 25.3, None),
    ("D", "ε_ls=0.2", 6, 512, 2048, 8, 64, 64, 0.1, 0.2, "100K", 5.47, 25.7, None),
    ("E", "learned positional embedding", 6, 512, 2048, 8, 64, 64, 0.1, 0.1, "100K", 4.92, 25.7, None),
    ("big", "big", 6, 1024, 4096, 16, 64, 64, 0.3, 0.1, "300K", 4.33, 26.4, 213),
]
GROUP_TEXT = {
    "A": "**Heads vs head size, compute held constant.** One head is 0.9 BLEU worse than the best; too many heads (32 × 16 dims) also hurts.",
    "B": "**Smaller keys hurt.** *\"Determining compatibility is not easy\"* — a richer compatibility function than the dot product might help.",
    "C": "**Bigger is better.** More layers, wider d_model and wider d_ff all improve dev perplexity and BLEU.",
    "D": "**Regularisation matters.** No dropout: −1.2 BLEU. No label smoothing: better perplexity, worse BLEU (tab ⑦).",
    "E": "**Sinusoids vs learned positions:** nearly identical (25.7 vs 25.8).",
}


def _param_count(N, d, dff, h, dk, dv, V) -> dict[str, float]:
    attn = 2 * (d * h * dk + h * dk) + (d * h * dv + h * dv) + (h * dv * d + d)
    ffn = d * dff + dff + dff * d + d
    enc = N * (attn + ffn + 2 * 2 * d)
    dec = N * (2 * attn + ffn + 3 * 2 * d)
    return {"Embeddings (shared)": V * d, "Encoder": enc, "Decoder": dec}


def _render_ablations() -> None:
    _owner("B")
    st.subheader("Table 3: what happens when you change one thing at a time (§6.2)")
    st.markdown(
        "**Protocol:** base EN→DE model, change one (group of) hyper-parameter(s), train 100k steps, evaluate on the **development set newstest2013** "
        "with beam search as in §6.1 but **without checkpoint averaging**. Perplexity is per word-piece (BPE token), so it is not comparable to per-word perplexities."
    )

    df = pd.DataFrame(TABLE_3, columns=["Group", "Change", "N", "d_model", "d_ff", "h", "d_k", "d_v", "P_drop", "ε_ls", "Steps",
                                        "PPL (dev)", "BLEU (dev)", "Params ×10⁶"])
    group = st.radio("Row group", ["All", "A", "B", "C", "D", "E"], horizontal=True, key="me_t3_group")
    view = df if group == "All" else df[df["Group"].isin(["base", group])]
    shown = view.drop(columns=["Group"]).copy()
    shown["Params ×10⁶"] = ["" if pd.isna(v) else f"{v:.0f}" for v in shown["Params ×10⁶"]]
    st.dataframe(shown, use_container_width=True, hide_index=True)
    if group in GROUP_TEXT:
        st.info(GROUP_TEXT[group])

    plot_df = view.copy()
    plot_df["label"] = [c if g in ("base", "big") else f"{g}: {c}" for g, c in zip(plot_df["Group"], plot_df["Change"])]
    colors = [AMBER if g == "base" else (INDIGO if g == "big" else BLUE) for g in plot_df["Group"]]
    l, r = st.columns(2)
    with l:
        fig = go.Figure(go.Bar(y=plot_df["label"], x=plot_df["BLEU (dev)"], orientation="h", marker_color=colors,
                               text=plot_df["BLEU (dev)"], textposition="outside", cliponaxis=False))
        fig.add_vline(x=25.8, line=dict(color=AMBER, dash="dot"))
        fig.update_layout(height=120 + 26 * len(plot_df), margin=dict(l=10, r=30, t=30, b=10), xaxis=dict(range=[23, 27]),
                          yaxis=dict(autorange="reversed"), title="BLEU on dev (higher = better; dotted = base)")
        st.plotly_chart(fig, use_container_width=True)
    with r:
        fig = go.Figure(go.Bar(y=plot_df["label"], x=plot_df["PPL (dev)"], orientation="h", marker_color=colors,
                               text=plot_df["PPL (dev)"], textposition="outside", cliponaxis=False))
        fig.add_vline(x=4.92, line=dict(color=AMBER, dash="dot"))
        fig.update_layout(height=120 + 26 * len(plot_df), margin=dict(l=10, r=30, t=30, b=10), xaxis=dict(range=[4, 6.4]),
                          yaxis=dict(autorange="reversed", showticklabels=False), title="Perplexity on dev (lower = better)")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### How the 'Params' column is generated")
    st.markdown("Parameters follow directly from the hyper-parameters. Pick a configuration and compare our count with the paper's:")
    c1, c2, c3, c4 = st.columns(4)
    sized = df.dropna(subset=["Params ×10⁶"])
    row_i = c1.selectbox("Configuration", sized.index, format_func=lambda i: df.loc[i, "Change"], key="me_t3_row")
    V = c2.number_input("Vocabulary size", 20000, 60000, 37000, 1000, key="me_t3_v")
    row = df.loc[row_i]
    parts = _param_count(int(row["N"]), int(row["d_model"]), int(row["d_ff"]), int(row["h"]), int(row["d_k"]), int(row["d_v"]), int(V))
    total = sum(parts.values()) / 1e6
    c3.metric("Our count", f"{total:.1f} M")
    c4.metric("Paper", f"{row['Params ×10⁶']:.0f} M", delta=f"{total - row['Params ×10⁶']:+.1f} M", delta_color="off")
    fig = go.Figure()
    for k, (name, v) in enumerate(parts.items()):
        fig.add_trace(go.Bar(x=[v / 1e6], y=["params"], orientation="h", name=name, marker_color=[AMBER, BLUE, GREEN][k],
                             text=f"{name}: {v / 1e6:.1f}M", textposition="inside"))
    fig.update_layout(barmode="stack", height=130, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="millions", yaxis=dict(visible=False),
                      xaxis=dict(range=[0, total * 1.02]),
                      legend=LEGEND)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Per attention block: Q, K projections d·h·d_k each, V projection d·h·d_v, output h·d_v·d (+ biases). Encoder layer = 1 attention + FFN; "
               "decoder layer = 2 attention + FFN. One embedding matrix shared three ways. With V ≈ 37k our counts land within ~4% of the paper; "
               "the paper only says 'about 37000', so the exact vocabulary (and any extra bias terms) is unknown.")

    st.markdown("#### Reading Table 3 critically")
    st.markdown(
        "- **One run per row**, no seeds or confidence intervals: differences of 0.1–0.3 BLEU may be noise.\n"
        "- **Rows (B) and (C) change model size too**, so 'better' is partly 'bigger'. Only rows (A) hold computation constant by design.\n"
        "- **Dev set only** — correct practice: hyper-parameters are chosen without looking at the test set.\n"
        "- The **big** row uses 300k steps and P_drop 0.3, so it differs from base in several ways at once."
    )

    _speaker_notes(
        "- Explain the protocol first: dev set, no checkpoint averaging. That's why base scores 25.8 here and 27.3 in Table 2.\n"
        "- Go group by group with the radio buttons; each group answers one design question (heads, key size, size, regularisation, positions).\n"
        "- Params demo: pick 'base' → ~63M vs 65M; pick 'big' → ~214M vs 213M. The column is computed, not measured.\n"
        "- End with the critical reading: one run each, no error bars. Strong conclusions only for large gaps (N=2, no dropout, h=1).\n"
        "- Bridge: *\"Last: does the model work beyond translation?\"*"
    )


# ----------------------------------------------------------------------------------------
# ⑩ Beyond translation: parsing setup + the 2026 paper's experimental setup
# ----------------------------------------------------------------------------------------
SWA_PROVENANCE = [
    ("Figure 1", "Attention masks: full, LoLCATs / Liger-GLA, SWA + sinks", "Diagram, adapted from Meng et al. (2026)"),
    ("Figure 2 (+ Fig. 3, 4 in appendix)", "Decoding speed, memory and FLOPs vs context length (128 → 256K)", "Measured by the authors on a 4-layer test model"),
    ("Table 1", "Summary of retrofitting methods: tokens, stages, MMLU, 6-task average, recovery %", "Mostly compiled from other papers + the authors' SWA row"),
    ("Table 2", "Short-context scores for 11 base models: teacher vs SWA(64, 4) vs linearised", "Measured by the authors"),
    ("Table 3", "GSM8K & HumanEval: SWA at several windows vs linear models", "Measured by the authors"),
    ("Tables 4–5", "Needle-in-a-Haystack and BABILong (QA1–5) up to 4K", "SWA measured; baselines copied from Meng et al. (2026)"),
    ("Table 6", "LongBench, LongBench-v2, RULER, BABILong at 8K / 16K", "Measured by the authors"),
    ("Table 7", "Authors' own retrofits (GLA, Gated DeltaNet, QRWKV6) on ~0.1B Alpaca tokens", "Trained and measured by the authors"),
]


def _render_beyond() -> None:
    _owner("B")
    st.subheader("English constituency parsing (§6.3): does it generalise?")
    st.markdown("A task where the output is **longer than the input and strongly structured**: the sentence goes in, a linearised parse tree comes out.")
    st.code("Input : The cat sat .\nOutput: (S (NP DT NN )NP (VP VBD )VP . )S      ← illustrative linearised tree", language=None)
    c1, c2 = st.columns(2)
    with c1:
        with st.container(border=True):
            st.markdown("**Setup**")
            st.markdown(
                "- 4-layer Transformer, d_model = 1024\n"
                "- WSJ-only: ≈ 40k sentences, 16k vocabulary\n"
                "- Semi-supervised: ≈ 17M sentences, 32k vocabulary\n"
                "- Max output length = input + 300\n"
                "- Beam size 21, α = 0.3"
            )
    with c2:
        with st.container(border=True):
            st.markdown("**What was tuned — and what wasn't**")
            st.markdown(
                "Only a *small number of experiments* on the Section 22 dev set, to select dropout (attention and residual), learning rates and beam size. "
                "**All other parameters stayed as in the EN→DE base model.** Evaluation: F1 on Section 23. "
                "This light tuning is the point: the same recipe transfers to a different task. Results → Results tab."
            )

    st.divider()
    st.subheader("The 2026 follow-up: how 'Sliding-Window Beats Linear Attention' ran its experiments")
    st.markdown(
        "**Question:** to cut the memory of a long KV cache in an existing LLM, is it better to *retrofit* it to linear attention "
        "(extra training) or just switch to **sliding-window attention (SWA) with attention sinks** (no training)? "
        "The mechanism itself is shown in **Code → Other Types Of Attention**; here: the experimental design."
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        with st.container(border=True):
            st.markdown("**Compared**")
            st.markdown("- Teacher (original model, full attention)\n- SWA with window 64 + 4 sinks (main), plus windows 128–512\n"
                        "- Retrofitted linear attention: SUPRA, Hedgehog, LoLCATs, Liger-GLA, MOHAWK, Mamba-in-Llama, DiJiang, ARWKV, Llamba, QRWKV6/7 …\n"
                        "- Hybrids with full-attention layers excluded")
    with c2:
        with st.container(border=True):
            st.markdown("**Models**")
            st.markdown("1.3B → 72B: Phi-1.5, Mistral-7B, Llama-2-7B, Llama-3/3.1-8B (+70B), Qwen2.5-7B/32B/72B, QwQ-32B. "
                        "Own retrofits of Qwen3-8B and Phi-4 reasoning models (~0.1B tokens of Alpaca).")
    with c3:
        with st.container(border=True):
            st.markdown("**Benchmarks & metrics**")
            st.markdown("- Short context (log-likelihood): MMLU, ARC-C/E, HellaSwag, PIQA, WinoGrande\n"
                        "- Generative: GSM8K (exact match), HumanEval (pass@1)\n"
                        "- Long context: Needle-in-a-Haystack, BABILong, LongBench(-v2), RULER up to 16K\n"
                        "- **Recovery** = student score ÷ teacher score")
    st.markdown("**Speed & memory:** a 4-layer Transformer (embedding 1024, 16 heads × 64), batch 1, float16, one RTX PRO 6000 Blackwell GPU, "
                "contexts 128 → 256K tokens; FlashAttention for full attention and SWA, ThunderKittens for linear attention.")

    st.markdown("#### Where each of its figures and tables comes from")
    st.dataframe(pd.DataFrame(SWA_PROVENANCE, columns=["Item", "Shows", "Source of the numbers"]), use_container_width=True, hide_index=True)

    st.markdown("#### Methodological notes")
    st.markdown(
        "- **Mixed provenance:** Table 1 and the long-context baselines (Tables 4–5) partly reuse numbers from other papers, "
        "so evaluation settings may differ between rows.\n"
        "- **Inconsistent token budgets** between tables (SUPRA: 100B in Table 1 vs 20B in Table 2; MOHAWK: 3–5B vs 3B).\n"
        "- **No stated reason** for the main window size of 64; larger windows are tested only on some tasks.\n"
        "- **Asymmetric comparison by design:** SWA needs zero training, linear methods 0.02B–40B+ tokens — that's the paper's point, "
        "but the result says nothing about linear models trained from scratch.\n"
        "- **Link to 2017:** SWA is the *restricted self-attention* (window r) that Vaswani et al. already listed in Table 1."
    )

    _speaker_notes(
        "- Parsing: emphasise how little was tuned; the claim is about **transfer of the recipe**.\n"
        "- 2026 paper: frame it as the same methodological questions applied to a new problem — which models, which baselines, which metrics, whose numbers.\n"
        "- The provenance table is the critical slide: some rows are measured by the authors, others copied.\n"
        "- Close the loop: the 'restricted attention' row of the 2017 Table 1 is exactly SWA.\n"
        "- Hand over to the Code section / Results."
    )


# ----------------------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------------------
def render_methodology():
    st.header("Methodology")
    st.caption(
        f"How the Transformer is built (Part A · {PRESENTERS['A']}) and how it was trained and evaluated (Part B · {PRESENTERS['B']}). ≈ 15 min."
    )

    tabs = st.tabs([
        "⓪ Overview",
        "① Architecture",
        "② Attention, Three Ways",
        "③ Word Order",
        "④ Attention Figures",
        "⑤ Data & Batching",
        "⑥ Optimizer & Schedule",
        "⑦ Regularization",
        "⑧ Decoding & Evaluation",
        "⑨ Model Variations",
        "⑩ Beyond Translation",
    ])
    renderers = [
        _render_overview,
        _render_architecture,
        _render_attention,
        _render_positions,
        _render_figures,
        _render_data,
        _render_training,
        _render_regularization,
        _render_decoding,
        _render_ablations,
        _render_beyond,
    ]
    for tab, render in zip(tabs, renderers):
        with tab:
            render()
