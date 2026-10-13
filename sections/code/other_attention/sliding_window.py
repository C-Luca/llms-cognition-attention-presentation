"""
========================================================================================
Module: Sliding Window Attention & Attention Sinks (The 2026 Paper)
Author: Cuclea Luca-Nicolae & Gheorghe Bogdan-Alexandru (Coder)
========================================================================================
"""

import numpy as np
import streamlit as st
import plotly.express as px

def render_sliding_window():
    st.header("Sliding Window Attention & Attention Sinks: The 2026 Paper")
    
    col_swa_left, col_swa_right = st.columns([1, 1])
    
    with col_swa_left:
        st.subheader("Memory Bottleneck: KV Cache Explosion")
        st.markdown(r"""
        During step-by-step autoregressive text generation (inference), we cache all previous $K$ and $V$ tensors in GPU memory (**KV Cache**):
        - **Full Attention:** Memory consumption grows linearly with sequence length $\mathcal{O}(N)$, while attention computation scales quadratically $\mathcal{O}(N^2)$.
        - At long contexts of 32k or 128k tokens, the KV Cache memory footprint easily exceeds the parameter size of the entire model!
        """)
        
        # Interactive KV Cache Calculator
        st.markdown("#### 🧮 Interactive KV Cache Memory Calculator")
        c_seq_len = st.select_slider("Context Length (N tokens)", options=[1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072], value=32768)
        c_layers = 32
        c_heads = 32
        c_head_dim = 128
        precision_bytes = 2 # FP16
        
        # KV Cache formula: 2 (K and V) * layers * heads * head_dim * seq_len * bytes
        kv_cache_full_gb = (2 * c_layers * c_heads * c_head_dim * c_seq_len * precision_bytes) / (1024**3)
        
        # Sliding Window with window=1024 + 4 sinks
        window_size = 1024
        sinks = 4
        effective_swa_tokens = min(c_seq_len, window_size + sinks)
        kv_cache_swa_gb = (2 * c_layers * c_heads * c_head_dim * effective_swa_tokens * precision_bytes) / (1024**3)
        
        m_c1, m_c2 = st.columns(2)
        m_c1.metric("Full Attention KV Cache", f"{kv_cache_full_gb:.2f} GB VRAM")
        m_c2.metric(f"SWA (Window {window_size} + Sinks)", f"{kv_cache_swa_gb:.2f} GB VRAM", delta=f"-{(1 - kv_cache_swa_gb/kv_cache_full_gb)*100:.1f}%")

    with col_swa_right:
        st.subheader("What is an 'Attention Sink'?")
        st.markdown(r"""
        If we simply discard old tokens using a naive sliding window (naive SWA), the model fails catastrophically and perplexity explodes.
        **Key discovery:** The first few tokens of a sequence (e.g., initial prompt tokens or `<s>` / BOS) absorb massive amounts of attention weight, regardless of their semantic relevance!
        They serve as an **attention sink** satisfying the softmax normalization constraint ($\sum = 1$).
        
        By persistently retaining **the first $S$ sink tokens** + **the most recent $W$ local tokens**, the model maintains stable language modeling performance indefinitely without any fine-tuning!
        """)
        
        # Mask Visualizer
        mask_type = st.radio("Attention Mask Type:", ["1. Full Causal Attention", "2. Naive Sliding Window (no sinks)", "3. SWA with Attention Sinks (2026 Paper)"], horizontal=True)
        
        n_vis = 16
        w_vis = 4
        s_vis = 2
        mask_mat = np.zeros((n_vis, n_vis))
        
        for i in range(n_vis):
            for j in range(n_vis):
                if j <= i: # causal condition
                    if mask_type.startswith("1"):
                        mask_mat[i, j] = 1.0
                    elif mask_type.startswith("2"):
                        if i - j < w_vis:
                            mask_mat[i, j] = 1.0
                    else:
                        if j < s_vis or (i - j < w_vis):
                            mask_mat[i, j] = 1.0
                            
        fig_mask = px.imshow(
            mask_mat,
            color_continuous_scale=[[0, "#0f172a"], [1, "#38bdf8"]],
            title=f"Attention Mask Matrix (16x16) for: {mask_type}",
            labels=dict(x="Key Index", y="Query Index")
        )
        fig_mask.update_layout(height=320, coloraxis_showscale=False)
        st.plotly_chart(fig_mask, use_container_width=True)

    st.divider()
    st.markdown("#### 💻 PyTorch Implementation for SWA + Attention Sinks:")
    st.code("""def sliding_window_with_sinks_mask(seq_len, window_size=1024, num_sinks=4):
    # 1. Lower-triangular causal mask
    mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
    
    # 2. Sliding band mask (tokens further than window_size become False)
    band_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=-(window_size - 1))
    swa_mask = mask & band_mask
    
    # 3. Retain the first num_sinks columns (Attention Sinks) active across the entire context!
    swa_mask[:, :num_sinks] = mask[:, :num_sinks]
    
    return swa_mask # Shape: [seq_len, seq_len]
""", language="python")
