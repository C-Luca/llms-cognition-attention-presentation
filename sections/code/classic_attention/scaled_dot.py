"""
========================================================================================
Module: Scaled Dot-Product & Softmax Saturation
Author: Cuclea Luca-Nicolae & Gheorghe Bogdan-Alexandru (Coder)
========================================================================================
"""

import math
import numpy as np
import torch
import streamlit as st
import plotly.express as px

def render_scaled_dot():
    st.header("Mathematical Laboratory: Scaled Dot-Product & Softmax Saturation")
    
    col_ctrl, col_plot = st.columns([1, 2])
    
    with col_ctrl:
        st.subheader("Experiment Parameters")
        d_k = st.slider("Key vector dimension ($d_k$)", min_value=16, max_value=512, value=64, step=16)
        seq_len = st.slider("Sequence length ($N$ tokens)", min_value=4, max_value=16, value=8, step=1)
        scale_toggle = st.toggle("Apply scaling by 1 / √dₖ (Standard Transformer)", value=True)
        mask_toggle = st.toggle("Apply Causal Mask (Autoregressive Decoder)", value=False)
        seed = st.number_input("Random Seed (for reproducibility)", value=42, step=1)
        
        torch.manual_seed(int(seed))
        q = torch.randn(1, 1, seq_len, d_k)
        k = torch.randn(1, 1, seq_len, d_k)
        
        raw_scores = torch.matmul(q, k.transpose(-2, -1))
        scale_factor = math.sqrt(d_k) if scale_toggle else 1.0
        scaled_scores = raw_scores / scale_factor
        
        if mask_toggle:
            causal_mask = torch.tril(torch.ones(seq_len, seq_len))
            scaled_scores = scaled_scores.masked_fill(causal_mask == 0, -1e9)
            
        attn_weights = torch.softmax(scaled_scores, dim=-1).squeeze().detach().numpy()
        scores_matrix = scaled_scores.squeeze().detach().numpy()
        
        st.divider()
        var_raw = float(raw_scores.var())
        var_scaled = float((raw_scores / math.sqrt(d_k)).var())
        st.metric(label="Raw score variance (Q · K^T)", value=f"{var_raw:.2f}")
        st.metric(label="Variance after scaling (/ √d_k)", value=f"{var_scaled:.2f}")
        
        if not scale_toggle:
            st.warning(f"Without scaling, at d_k={d_k}, the variance reaches {var_raw:.1f}! The softmax saturates into extreme one-hot distributions (near 0 and 1), causing vanishing gradients!")

    with col_plot:
        st.subheader("Resulting Attention Matrix (Interactive Heatmap)")
        token_labels = [f"Token_{i}" for i in range(seq_len)]
        
        fig = px.imshow(
            attn_weights,
            labels=dict(x="Key Token (K)", y="Query Token (Q)", color="Attention"),
            x=token_labels,
            y=token_labels,
            color_continuous_scale="Viridis",
            zmin=0.0,
            zmax=1.0,
            title=f"Softmax Attention Matrix (Scaling: {'ENABLED (1/√' + str(d_k) + ')' if scale_toggle else 'DISABLED'})"
        )
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)
