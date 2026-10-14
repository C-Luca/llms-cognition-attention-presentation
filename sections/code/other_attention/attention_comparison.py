"""
Streamlit Application: Attention Mechanisms Comparison
Visual demonstration comparing Classic Causal Attention vs. Sliding Window Attention.
Can be executed directly via:
    uv run streamlit run sections/code/other_attention/attention_comparison.py
"""

import streamlit as st
from sections.code.other_attention.sliding_window import render_sliding_window

if __name__ == "__main__":
    st.set_page_config(
        page_title="Attention Mechanisms: Causal vs. Sliding Window",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    render_sliding_window()
