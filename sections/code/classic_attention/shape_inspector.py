"""
========================================================================================
Module: PyTorch Shape Inspector & Live Forward Pass
Author: Cuclea Luca-Nicolae & Gheorghe Bogdan-Alexandru (Coder)
========================================================================================
"""

import math
import torch
import torch.nn as nn
import streamlit as st

def scaled_dot_product_attention(q, k, v, mask=None, dropout=None):
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    attn_weights = torch.softmax(scores, dim=-1)
    if dropout is not None:
        attn_weights = dropout(attn_weights)
    output = torch.matmul(attn_weights, v)
    return output, attn_weights

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=512, num_heads=8, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        q = self.w_q(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.w_k(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.w_v(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        context, attn_weights = scaled_dot_product_attention(
            q, k, v, mask=mask, dropout=self.dropout
        )
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.w_o(context)
        return output, attn_weights

def render_shape_inspector():
    st.header("PyTorch Shape Inspector & Live Forward Pass")
    
    f_c1, f_c2, f_c3, f_c4 = st.columns(4)
    with f_c1:
        f_batch = st.number_input("Batch Size (B)", min_value=1, max_value=8, value=2)
    with f_c2:
        f_seq = st.number_input("Sequence Length (N)", min_value=2, max_value=64, value=11)
    with f_c3:
        f_dmodel = st.selectbox("d_model (Model Dimension)", options=[256, 512, 768, 1024], index=1)
    with f_c4:
        f_heads = st.selectbox("num_heads (h)", options=[2, 4, 8, 16], index=2)
        
    f_dk = f_dmodel // f_heads
    st.caption(f"Dimension of each individual attention head ($d_k = d_{{model}} / h$): **{f_dk}**")
    
    if st.button("Run Live Forward Pass on GPU/CPU", type="primary"):
        model = MultiHeadAttention(d_model=f_dmodel, num_heads=f_heads)
        x = torch.randn(f_batch, f_seq, f_dmodel)
        
        output, weights = model(x, x, x)
        
        st.success("Forward Pass completed successfully with zero errors!")
        
        s1, s2, s3 = st.columns(3)
        with s1:
            st.markdown("#### 1. Input Tensor")
            st.code(f"{list(x.shape)}\n[Batch, Seq_Len, d_model]")
        with s2:
            st.markdown("#### 2. Attention Weights")
            st.code(f"{list(weights.shape)}\n[Batch, Heads, Seq_Len, Seq_Len]")
        with s3:
            st.markdown("#### 3. Output Tensor")
            st.code(f"{list(output.shape)}\n[Batch, Seq_Len, d_model]")
            
        st.info(f"""
        **Hardware-Level Insights:**
        - The attention weights tensor `weights` contains exactly **{f_batch * f_heads}** parallel matrices of dimension `{f_seq} x {f_seq}`.
        - Because the head dimension `h` is permuted forward with `.transpose(1, 2)`, all {f_heads} attention heads are computed simultaneously as a single batched matrix multiplication!
        """)
