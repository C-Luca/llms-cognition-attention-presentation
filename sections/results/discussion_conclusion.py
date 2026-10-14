"""
========================================================================================
Module: Results - Discussion & Conclusion
Author: Gusev Roman (Results Interpreter & Challenger)
========================================================================================
Paper Sec. 7 (Conclusion) and what happened to the future work.
========================================================================================
"""

import streamlit as st

from sections.results.common import md_table, speaker_notes

FUTURE = [
    ("Other modalities: images, audio, video",
     "Done at scale", "Vision Transformer (Dosovitskiy et al., 2020) for images; Transformer encoder-decoder speech models such as Whisper (2022)."),
    ("Local, restricted attention for large inputs",
     "Done: and in our Code section", "Table 1's 'restricted to r neighbours' row returns as sliding-window attention (Longformer 2020; the 2026 paper in **Code $\\rightarrow$ Other Types Of Attention**)."),
    ("Make generation less sequential",
     "Partly", "Decoding is still one token at a time in today's LLMs; non-autoregressive and speculative decoding reduce but do not remove it."),
]


def render_discussion_conclusion():
    tabs = st.tabs(["Conclusion in Numbers", "Future Work: What Happened?"])

    # ---- 1
    with tabs[0]:
        st.subheader("The paper's conclusion")
        st.markdown(
            "> *\"...the Transformer, the first sequence transduction model based entirely on attention, replacing the recurrent layers most commonly used "
            "in encoder-decoder architectures with multi-headed self-attention.\"*"
        )
        m1, m2, m3 = st.columns(3)
        m1.metric("EN-DE (big)", "28.4 BLEU", delta="+2.04 vs best ensemble")
        m2.metric("EN-FR (big)", "41.8 BLEU", delta="+0.51 vs best ensemble", help="Sec. 6.1 text says 41.0; Table 2 says 41.8.")
        m3.metric("Training cost (big)", "2.3e19 FLOPs", delta="3.5 days, 8 P100", delta_color="off")

        c1, c2 = st.columns(2)
        with c1:
            with st.container(border=True):
                st.markdown("**1. Quality**")
                st.markdown("New state of the art on both WMT 2014 tasks; on EN-DE it beats even all earlier *ensembles*.")
        with c2:
            with st.container(border=True):
                st.markdown("**2. Speed**")
                st.markdown("Trains significantly faster than recurrent or convolutional models: no sequential steps inside a sentence.")

    # ---- 2
    with tabs[1]:
        st.subheader("The authors' plans and what became of them")
        st.markdown("The conclusion lists three directions. Nine years on:")
        md_table(["Plan in the paper", "Status", "What happened"], FUTURE)
        st.success(
            "The conclusion's quiet last point, 'restricted attention for large inputs', is the thread that links this paper to the 2026 article in our Code section: "
            "the $O(n^2)$ price highlighted in Table 1 is exactly what sliding-window attention tries to avoid."
        )
