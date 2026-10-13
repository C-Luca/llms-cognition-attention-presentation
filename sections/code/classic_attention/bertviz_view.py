"""
========================================================================================
Module: BertViz Multiscale Attention Explorer
Author: Cuclea Luca-Nicolae & Gheorghe Bogdan-Alexandru (Coder)
Paper Reference: "A Multiscale Visualization of Attention in the Transformer Model" (Jesse Vig, ACL 2019)
========================================================================================
"""

import os
import streamlit as st
import streamlit.components.v1 as components

ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")

@st.cache_data
def get_view_html(filename: str) -> str:
    file_path = os.path.join(ASSETS_DIR, filename)
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    return f"<p style='color:red;'>Asset {filename} not found in {ASSETS_DIR}.</p>"


def render_bertviz_view():
    st.header("Multiscale Attention Explorer (BertViz)")
    st.markdown(r"""
    **BertViz** (Jesse Vig, ACL 2019) is the academic gold standard for inspecting the inner mechanisms of Transformer attention.
    Under the hood, it extracts the multi-head attention tensors from `bert-base-uncased` ($L=12$ layers, $H=12$ heads, $d_{model}=768$, $d_k=64$, totalling **144 attention heads**) 
    evaluated on the sentence pair:
    - **Sentence A:** `The cat sat on the mat`
    - **Sentence B:** `The cat lay on the rug`
    """)

    selected_view = st.radio(
        "Select Multiscale Attention View:",
        [
            "👁️ 1. Head View (Bipartite Token Graph & Head Specialization)",
            "🌐 2. Model View (Bird's-Eye Grid across all 144 Attention Heads)",
            "🔬 3. Neuron View (Microscopic Query-Key Dot Product Tracing)"
        ],
        horizontal=True
    )

    st.divider()

    if selected_view.startswith("👁️ 1."):
        col_view, col_info = st.columns([3, 2])
        with col_view:
            st.subheader("1. Head View (Single Layer)")
            st.caption("Interactive bipartite graph: hover tokens to filter connections; click colored tiles to isolate heads.")
            html_content = get_view_html("head_view.html")
            components.html(html_content, height=520, scrolling=True)
            
        with col_info:
            st.markdown("### 📊 Presentation & Architectural Guide")
            st.markdown("""
            **What the visualization displays:**
            - Connects each Query token (left) to each Key token (right).
            - Line **thickness & opacity** represents the softmax attention weight $\\alpha_{ij} \\in [0, 1]$.
            - Line **color** corresponds to the specific attention head.

            **Key Insights for Class Presentation:**
            1. **Head Specialization (No Redundancy):**
               - Notice that different heads learn distinct grammatical roles.
               - Some heads bind adjacent tokens (e.g., determiner *'The'* to noun *'cat'*), while other heads capture long-range verbal arguments (*'sat'* to prepositional phrase *'on the mat'*).
            2. **Cross-Sentence Semantic Alignment:**
               - Look at the interaction between Sentence A and Sentence B: tokens like *'mat'* and *'rug'* exhibit direct mutual attention, illustrating how self-attention matches semantically equivalent concepts across distinct clauses.
            3. **Interactive Demonstration:**
               - *Double-click* any head tile at the top to isolate a single head.
               - *Hover* over `cat` or `mat` to highlight incoming and outgoing attention paths.
            """)

    elif selected_view.startswith("🌐 2."):
        col_view, col_info = st.columns([3, 2])
        with col_view:
            st.subheader("2. Model View (Global 12x12 Grid)")
            st.caption("Bird's-eye view across all 12 layers (rows) and 12 heads (columns). Click any cell to inspect.")
            html_content = get_view_html("model_view.html")
            components.html(html_content, height=620, scrolling=True)
            
        with col_info:
            st.markdown("### 📊 Presentation & Architectural Guide")
            st.markdown(r"""
            **What the visualization displays:**
            - A full macroscopic grid of all **144 attention heads** ($12 \text{ layers} \times 12 \text{ heads}$).
            - Each cell depicts the miniature attention pattern between all tokens in the sequence.

            **Critical Insights for Class Presentation:**
            1. **Layer Depth Hierarchy:**
               - **Lower Layers (0–3):** Primarily local and positional attention (strong diagonal patterns focusing on the previous or next token).
               - **Middle Layers (4–8):** Syntactic parsing, long-range dependencies, and cross-sentence coreference tracking.
               - **Higher Layers (9–11):** Global semantic integration and task-oriented representations.
            2. **The Direct Bridge to Attention Sinks (2026 Paper):**
               - *Notice how in deeper layers, multiple heads focus overwhelmingly on `[CLS]` and `[SEP]`!*
               - When a token does not require external context, the softmax normalization constraint ($\sum \alpha_{ij} = 1$) forces it to allocate probability mass somewhere.
               - The model learns to dump excess attention into delimiter tokens — providing empirical proof for the **Attention Sink phenomenon** covered in our second paper!
            """)

    else:
        col_view, col_info = st.columns([3, 2])
        with col_view:
            st.subheader("3. Neuron View (Vector Dot Product Tracing)")
            st.caption("Hover over a left token, then click the '+' icon to expose query/key neurons.")
            html_content = get_view_html("neuron_view.html")
            components.html(html_content, height=620, scrolling=True)
            
        with col_info:
            st.markdown("### 📊 Presentation & Architectural Guide")
            st.markdown(r"""
            **What the visualization displays:**
            - Traces the exact computation from individual neuron activations ($d_k = 64$) up to the final softmax score.
            - **Blue bands** represent positive neuron values; **Orange bands** represent negative values. Color intensity reflects magnitude.

            **Mathematical Insights for Class Presentation:**
            1. **Anatomy of the Dot Product:**
               - Shows how query vector $q \in \mathbb{R}^{d_k}$ and key vector $k \in \mathbb{R}^{d_k}$ interact component by component ($q_i \cdot k_i$).
               - If two neurons have the same sign (both blue or both orange), their product is positive, directly increasing attention.
            2. **Why Scaling by $1/\sqrt{d_k}$ Matters:**
               - Summing across all $64$ neuron products yields large unnormalized scalar logits.
               - Dividing by $\sqrt{64} = 8$ prevents the dot products from pushing the softmax into vanishing gradient saturation regions!
            """)
