"""
========================================================================================
Modul: Scaled Dot-Product & Saturație
Responsabil: Cuclea Luca-Nicolae & Gheorghe Bogdan-Alexandru (Coder)
========================================================================================
"""

import math
import numpy as np
import torch
import streamlit as st
import plotly.express as px

def render_scaled_dot():
    st.header("Laborator Matematic: Scaled Dot-Product & Fenomenul Saturației Softmax")
    
    col_ctrl, col_plot = st.columns([1, 2])
    
    with col_ctrl:
        st.subheader("Parametri Experiment")
        d_k = st.slider("Dimensiunea vectorului cheie ($d_k$)", min_value=16, max_value=512, value=64, step=16)
        seq_len = st.slider("Lungimea secvenței ($N$ tokeni)", min_value=4, max_value=16, value=8, step=1)
        scale_toggle = st.toggle("Aplică scalarea cu 1 / √dₖ (Standard Transformer)", value=True)
        mask_toggle = st.toggle("Aplică Causal Mask (Decoder Autoregresiv)", value=False)
        seed = st.number_input("Random Seed (pentru reproductibilitate)", value=42, step=1)
        
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
        st.metric(label="Varianța scorurilor brute (Q · K^T)", value=f"{var_raw:.2f}")
        st.metric(label="Varianța după scalare (/ √d_k)", value=f"{var_scaled:.2f}")
        
        if not scale_toggle:
            st.warning(f"⚠️ Fără scalare, la d_k={d_k}, varianța este {var_raw:.1f}! Softmax-ul se saturează în valori de 0 și 1 (one-hot), distrugând gradientul (Vanishing Gradient)!")

    with col_plot:
        st.subheader("Matricea de Atenție Rezultată (Heatmap Interactiv)")
        token_labels = [f"Token_{i}" for i in range(seq_len)]
        
        fig = px.imshow(
            attn_weights,
            labels=dict(x="Key Token (K)", y="Query Token (Q)", color="Atenție"),
            x=token_labels,
            y=token_labels,
            color_continuous_scale="Viridis",
            zmin=0.0,
            zmax=1.0,
            title=f"Matricea Softmax (Scalare: {'ACTIVATĂ (1/√' + str(d_k) + ')' if scale_toggle else 'DEZACTIVATĂ'})"
        )
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)
