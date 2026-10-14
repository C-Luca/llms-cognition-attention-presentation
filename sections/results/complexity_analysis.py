"""
========================================================================================
Module: Results - Complexity Analysis (paper Sec. 4 "Why Self-Attention", Table 1)
Author: Gusev Roman (Results Interpreter & Challenger)
========================================================================================
The theoretical argument behind the measured results: three criteria, Table 1, and where
each entry comes from.
========================================================================================
"""

import streamlit as st

from sections.results.common import md_table, speaker_notes

# Table 1 of the paper, verbatim in meaning: (layer type, complexity per layer, sequential operations, maximum path length)
TABLE_1 = [
    ("Self-attention", "$O(n^2 \\cdot d)$", "$O(1)$", "$O(1)$"),
    ("Recurrent", "$O(n \\cdot d^2)$", "$O(n)$", "$O(n)$"),
    ("Convolutional", "$O(k \\cdot n \\cdot d^2)$", "$O(1)$", "$O(\\log_k n)$"),
    ("Self-attention (restricted)", "$O(r \\cdot n \\cdot d)$", "$O(1)$", "$O(n/r)$"),
]


# ----------------------------------------------------------------------------------------
# 1. The three criteria
# ----------------------------------------------------------------------------------------
def _render_criteria() -> None:
    st.subheader("Why self-attention? Three things a layer should minimise")
    st.markdown(
        "The paper compares **layer types** that all do the same job: map a sequence of $n$ vectors "
        "$(x_1, \\dots, x_n)$, each of size $d$, to another sequence of the same length. The authors judge them on three criteria:"
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        with st.container(border=True):
            st.caption("Criterion 1")
            st.markdown("**Computation per layer**")
            st.markdown("How many arithmetic operations one layer needs for a sequence of length $n$ with vectors of size $d$. Lower means cheaper to train and run.")
    with c2:
        with st.container(border=True):
            st.caption("Criterion 2")
            st.markdown("**Sequential operations**")
            st.markdown("The minimum number of steps that must happen *one after another*. Everything else can run in parallel on a GPU, so fewer sequential steps means faster training.")
    with c3:
        with st.container(border=True):
            st.caption("Criterion 3")
            st.markdown("**Maximum path length**")
            st.markdown("How many steps a signal must travel between the two *furthest* positions. Short paths make long-range dependencies easier to learn (Hochreiter et al., 2001).")

    st.markdown("#### Notation used in Table 1")
    md_table(
        ["Symbol", "Meaning", "Typical value"],
        [
            ("$n$", "Sequence length (number of tokens)", "a sentence: tens of tokens; a long document: thousands"),
            ("$d$", "Representation size ($d_{\\text{model}}$ in the Transformer)", "512 (base), 1024 (big)"),
            ("$k$", "Kernel width of a convolution", "3 to 7"),
            ("$r$", "Neighbourhood size in restricted self-attention", "chosen by the designer"),
        ],
    )

    # speaker_notes(
    #     "- Three yardsticks: how much work, how much of it must wait, how far apart two words are in the network.\n"
    #     "- Sequential operations decide speed on GPUs; path length decides how easily distant words influence each other.\n"
    #     "- Define n and d once: n = number of tokens, d = vector size. Everything in Table 1 is a function of these.\n"
    #     "- The paper's own condition: self-attention is cheaper than recurrence when n < d.\n"
    #     "- Bridge: *\"Here is the table, and where each entry comes from.\"*"
    # )


# ----------------------------------------------------------------------------------------
# 2. Table 1 explained
# ----------------------------------------------------------------------------------------
def _render_table() -> None:
    st.subheader("Complexity, sequential operations and path length")
    md_table(["Layer type", "Complexity per layer", "Sequential operations", "Maximum path length"], TABLE_1)
    st.caption("Table 1. $n$ = sequence length, $d$ = representation size, $k$ = kernel width, $r$ = neighbourhood size.")

    st.markdown("#### Where each row comes from")
    a, b = st.columns(2)
    with a:
        with st.container(border=True):
            st.markdown("**Self-attention**")
            st.markdown(
                "- The score matrix $QK^{T}$ has $n \\times n$ entries, each a dot product of $d$ numbers: **$n^2 \\cdot d$** operations. "
                "Weighting the values costs the same.\n"
                "- One matrix multiplication handles every position: **$O(1)$** sequential steps.\n"
                "- Every position reads every other directly: path length **$O(1)$**."
            )
        with st.container(border=True):
            st.markdown("**Convolutional**")
            st.markdown(
                "- Each of $n$ outputs mixes $k$ neighbours, each a $d \\to d$ transformation: **$k \\cdot n \\cdot d^2$**.\n"
                "- Outputs are independent: **$O(1)$** sequential steps.\n"
                "- One layer sees only $k$ positions. Linking all pairs needs $O(n/k)$ stacked layers with contiguous kernels, "
                "or **$O(\\log_k n)$** with dilated kernels (the value in the table)."
            )
    with b:
        with st.container(border=True):
            st.markdown("**Recurrent**")
            st.markdown(
                "- Each step multiplies a $d$-sized state by a $d \\times d$ matrix: $d^2$ per word, $n$ words: **$n \\cdot d^2$**.\n"
                "- Step $t$ needs step $t-1$: **$O(n)$** sequential steps.\n"
                "- Word 1 reaches word $n$ only through every word in between: path length **$O(n)$**."
            )
        with st.container(border=True):
            st.markdown("**Self-attention, restricted to $r$ neighbours**")
            st.markdown(
                "- Each position attends to only $r$ others: **$r \\cdot n \\cdot d$**.\n"
                "- Still **$O(1)$** sequential steps.\n"
                "- Distant words are linked through chains of neighbourhoods: path **$O(n/r)$**. "
                "The authors propose it for very long inputs as future work."
            )


# ----------------------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------------------
def render_complexity_analysis():
    st.header("Complexity Analysis")

    tabs = st.tabs(["📐 Comparison Criteria", "🧮 Complexities Comparison"])
    for tab, render in zip(tabs, [_render_criteria, _render_table]):
        with tab:
            render()
