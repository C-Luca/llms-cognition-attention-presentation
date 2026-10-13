"""
========================================================================================
Module: Code Comparison: 3 Paradigms
Author: Cuclea Luca-Nicolae & Gheorghe Bogdan-Alexandru (Coder)
========================================================================================
"""

import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import streamlit as st

class PureNumpyMultiHeadAttention:
    def __init__(self, d_model=64, num_heads=4):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Manually initialized mock weights
        rng = np.random.RandomState(42)
        self.W_q = rng.randn(d_model, d_model) * 0.02
        self.W_k = rng.randn(d_model, d_model) * 0.02
        self.W_v = rng.randn(d_model, d_model) * 0.02
        self.W_o = rng.randn(d_model, d_model) * 0.02

    def forward(self, x, is_causal=True):
        N, D = x.shape
        # 1. Linear projections: [N, D] @ [D, D] -> [N, D]
        Q = np.dot(x, self.W_q)
        K = np.dot(x, self.W_k)
        V = np.dot(x, self.W_v)
        
        # 2. Reshape & Split into heads: [num_heads, N, d_k]
        Q = Q.reshape(N, self.num_heads, self.d_k).swapaxes(0, 1)
        K = K.reshape(N, self.num_heads, self.d_k).swapaxes(0, 1)
        V = V.reshape(N, self.num_heads, self.d_k).swapaxes(0, 1)
        
        # 3. Scaled Dot-Product Attention
        scores = np.matmul(Q, K.swapaxes(-2, -1)) / math.sqrt(self.d_k)
        
        # 4. Causal Mask (upper triangular -> -1e9)
        if is_causal:
            causal_mask = np.tril(np.ones((N, N)))
            scores = np.where(causal_mask == 1, scores, -1e9)
            
        # 5. Numerically stable manual softmax along the last axis
        exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        if is_causal:
            exp_scores = np.where(causal_mask == 1, exp_scores, 0.0)
        attn_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
        
        # 6. Weighted context: [heads, N, N] @ [heads, N, d_k] -> [heads, N, d_k]
        context = np.matmul(attn_weights, V)
        
        # 7. Concatenate heads back: [N, D]
        context = context.swapaxes(0, 1).reshape(N, D)
        
        # 8. Output projection
        output = np.dot(context, self.W_o)
        return output, attn_weights


class ModernFlashAttention(nn.Module):
    def __init__(self, d_model=64, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, is_causal=True):
        B, N, D = x.shape
        qkv = self.qkv_proj(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        out = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
        out = out.transpose(1, 2).reshape(B, N, D)
        return self.out_proj(out)


def render_code_paradigms():
    st.header("Syntax Comparison: 3 Ways to Implement the Same Attention Mechanism")
    st.markdown(r"""
    All 3 code implementations below execute the **exact same mathematical computation**:
    a **Multi-Head Self-Attention** module with linear projections $Q, K, V$, scaling by $\sqrt{d_k}$, optional causal masking, softmax normalization, and output projection $W_O$.
    Notice the evolution of syntax, boilerplate reduction, and abstraction layers from 2017 to 2026.
    """)

    code_paradigm = st.radio(
        "Select code paradigm for detailed analysis:",
        [
            "🐍 1. Pure Python & NumPy (From Scratch - No ML Libraries)",
            "🏛️ 2. Google Tensor2Tensor 2017 (Official Vaswani et al. Codebase)",
            "⚡ 3. PyTorch 2.x & Hugging Face (Modern 2026 Standard + FlashAttention)"
        ],
        horizontal=True
    )

    st.divider()

    if code_paradigm.startswith("🐍 1."):
        col_c1, col_c2 = st.columns([3, 2])
        with col_c1:
            st.subheader("1. Pure Python & NumPy (Explicit Mathematics)")
            st.caption("Full line-by-line implementation without any deep learning framework:")
            st.code(r"""import math
import numpy as np

class PureNumpyMultiHeadAttention:
    def __init__(self, d_model=64, num_heads=4):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Manually initialized mock weights
        rng = np.random.RandomState(42)
        self.W_q = rng.randn(d_model, d_model) * 0.02
        self.W_k = rng.randn(d_model, d_model) * 0.02
        self.W_v = rng.randn(d_model, d_model) * 0.02
        self.W_o = rng.randn(d_model, d_model) * 0.02

    def forward(self, x, is_causal=True):
        # x shape: [seq_len, d_model]
        N, D = x.shape
        
        # 1. Linear projections: [N, D] @ [D, D] -> [N, D]
        Q = np.dot(x, self.W_q)
        K = np.dot(x, self.W_k)
        V = np.dot(x, self.W_v)
        
        # 2. Reshape & Split into heads: [num_heads, N, d_k]
        Q = Q.reshape(N, self.num_heads, self.d_k).swapaxes(0, 1)
        K = K.reshape(N, self.num_heads, self.d_k).swapaxes(0, 1)
        V = V.reshape(N, self.num_heads, self.d_k).swapaxes(0, 1)
        
        # 3. Scaled Dot-Product Attention
        # MatMul: [heads, N, d_k] @ [heads, d_k, N] -> [heads, N, N]
        scores = np.matmul(Q, K.swapaxes(-2, -1)) / math.sqrt(self.d_k)
        
        # 4. Causal Mask (upper triangular -> -1e9)
        if is_causal:
            causal_mask = np.tril(np.ones((N, N)))
            scores = np.where(causal_mask == 1, scores, -1e9)
            
        # 5. Numerically stable manual softmax along the last axis
        exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        if is_causal:
            exp_scores = np.where(causal_mask == 1, exp_scores, 0.0)
        attn_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
        
        # 6. Weighted context: [heads, N, N] @ [heads, N, d_k] -> [heads, N, d_k]
        context = np.matmul(attn_weights, V)
        
        # 7. Concatenate heads back: [N, D]
        context = context.swapaxes(0, 1).reshape(N, D)
        
        # 8. Output projection
        output = np.dot(context, self.W_o)
        return output, attn_weights
""", language="python")

        with col_c2:
            st.markdown("### 📊 Characteristics & Analysis")
            st.markdown("""
            - **Lines of code:** ~45 lines.
            - **Dependencies:** Only `numpy` and `math`.
            - **Transparency:** 100% pedagogical — every tensor operation and mathematical formula is fully transparent.
            - **Drawbacks:**
              - Lack of automatic GPU acceleration (no optimized CUDA kernels).
              - Softmax requires manual subtraction of maximum values (`scores - max(scores)`) to prevent floating-point overflow.
              - No automatic differentiation (backward pass must be derived and implemented manually for gradient descent).
            """)
            
            if st.button("▶️ Run NumPy Code Live", key="run_numpy_mod"):
                x_test = np.random.randn(5, 64)
                module = PureNumpyMultiHeadAttention(d_model=64, num_heads=4)
                out_np, w_np = module.forward(x_test, is_causal=True)
                st.success(f"✓ Execution successful! Output shape: {out_np.shape}, Weights shape: {w_np.shape}")
                st.caption(f"Causality check: Attention weight to future tokens: {w_np[0, 0, 1]:.1e} (strictly 0)")

    elif code_paradigm.startswith("🏛️ 2."):
        col_c1, col_c2 = st.columns([3, 2])
        with col_c1:
            st.subheader("2. Official 2017 Codebase: Tensor2Tensor (TF 1.x)")
            st.caption("Actual production code written by Vaswani et al. in Google Brain's repository:")
            st.code(r"""# Adapted snippet from google/tensor2tensor/layers/common_attention.py
import tensorflow as tf
from tensor2tensor.layers import common_attention

def tensor2tensor_self_attention(x, num_heads=4, is_causal=True):
    # x shape: [batch_size, length, d_model]
    d_model = x.get_shape().as_list()[-1]
    length = tf.shape(x)[1]
    
    # 1. In T2T, the causal mask was represented as a 4D additive bias tensor:
    # Shape [1, 1, length, length] filled with -1e9 on the upper triangle
    if is_causal:
        bias = common_attention.attention_bias_lower_triangle(length)
    else:
        bias = None
        
    # 2. T2T library call projected Q, K, V and performed scaled attention
    # Note: Required constructing a static computational graph (TF 1.x)
    output = common_attention.multihead_attention(
        query_antecedent=x,
        memory_antecedent=None, # None specifies Self-Attention
        bias=bias,
        total_key_depth=d_model,
        total_value_depth=d_model,
        output_depth=d_model,
        num_heads=num_heads,
        dropout_rate=0.0,
        attention_type="dot_product",
        name="multihead_attention"
    )
    
    # 3. Execution required a TensorFlow session:
    # with tf.Session() as sess:
    #     out_val = sess.run(output, feed_dict={...})
    return output
""", language="python")

        with col_c2:
            st.markdown("### 📊 Characteristics & Analysis")
            st.markdown(r"""
            - **Lines of code:** ~30 lines in the caller script, but the underlying helper module (`common_attention.py`) contained **over 5,000 lines** of internal boilerplate!
            - **Dependencies:** `tensorflow==1.15.0`, `tensor2tensor`.
            - **2017 Architectural Peculiarities:**
              - The causal mask was not a boolean mask, but an **additive bias tensor** (`logits += bias`), where bias contained $-\infty$ for future positions and $0$ for past positions.
              - Static computational graph: tensors held no immediate numerical values in the console; they had to be compiled and evaluated inside a `tf.Session()`.
            - **Why it is obsolete today:**
              - Completely incompatible with modern GPU architectures and modern Python 3.10+ environments.
              - Heavy monolithic abstractions have been replaced by modular, eager-first frameworks (PyTorch).
            """)

    else:
        col_c1, col_c2 = st.columns([3, 2])
        with col_c1:
            st.subheader("3. PyTorch 2.x & Hugging Face (Modern 2026 Standard)")
            st.caption("Contemporary implementation utilizing CUDA kernel optimizations (FlashAttention):")
            st.code(r"""import torch
import torch.nn as nn
import torch.nn.functional as F

class ModernFlashAttention(nn.Module):
    def __init__(self, d_model=64, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        # Single fused linear projection for Q, K, V (saves GPU memory & kernel launches)
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, is_causal=True):
        # x shape: [batch, seq_len, d_model]
        B, N, D = x.shape
        
        # 1. Combined QKV projection in a single call: [B, N, 3, heads, head_dim]
        qkv = self.qkv_proj(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4) # -> [3, B, heads, N, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # 2. PyTorch 2.0+ Native Scaled Dot-Product Attention
        # Automatically invokes FlashAttention-2 or Memory-Efficient Attention in CUDA!
        # Fuses scaling, causal masking, and softmax into a single kernel!
        out = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
        
        # 3. Concatenation and output projection
        out = out.transpose(1, 2).reshape(B, N, D)
        return self.out_proj(out)
""", language="python")

        with col_c2:
            st.markdown("### 📊 Characteristics & Analysis")
            st.markdown(r"""
            - **Lines of code:** **16 lines** (massive reduction in boilerplate!).
            - **Key innovation:** `F.scaled_dot_product_attention(q, k, v, is_causal=True)` (standardized in PyTorch 2.0+).
            - **Massive hardware benefits:**
              - **FlashAttention-2 Fused Kernel:** Never materializes the intermediate $N \times N$ matrix in global GPU VRAM; computes softmax via *tiling* in ultra-fast SRAM!
              - Memory footprint drops from $\mathcal{O}(N^2)$ to $\mathcal{O}(N)$ during execution!
              - 2x–4x wall-clock speedup compared to standard attention implementations.
            """)
            
            if st.button("▶️ Run Modern PyTorch Module Live", key="run_pytorch_mod"):
                x_torch = torch.randn(2, 5, 64)
                mod_torch = ModernFlashAttention(d_model=64, num_heads=4)
                out_torch = mod_torch(x_torch, is_causal=True)
                st.success(f"✓ PyTorch 2.x execution successful! Output tensor shape: {list(out_torch.shape)}")
                st.caption("Causal attention was computed natively inside PyTorch without any manual mask allocation.")

    st.divider()
    st.subheader("Synoptic Comparison Across the 3 Paradigms")
    summary_data = {
        "Criterion": [
            "Lines of Code Written",
            "Execution Paradigm",
            "Causal Masking",
            "Softmax Implementation",
            "VRAM Memory Footprint",
            "Hardware Acceleration",
            "Where is it used today?"
        ],
        "1. Pure Python / NumPy": [
            "~45 lines",
            "Eager (CPU)",
            "Manual (`np.where` & `np.tril`)",
            "Manual with `np.exp(s - max(s))`",
            "High (allocates N x N matrix)",
            "No GPU / Slow SIMD",
            "Pedagogical / Technical Interviews"
        ],
        "2. Google Tensor2Tensor (2017)": [
            "~30 lines (+5,000 lines lib)",
            "Static Graph (`tf.Session`)",
            "4D Additive Bias (`+ bias`)",
            "Standard `tf.nn.softmax`",
            r"Quadratic $\mathcal{O}(N^2)$",
            "Basic NVIDIA CUDA",
            "Historical Archive (Deprecated)"
        ],
        "3. PyTorch 2.x Modern (2026)": [
            "~16 lines",
            "Eager / `torch.compile`",
            "Native parameter `is_causal=True`",
            "Fused CUDA kernel",
            r"Linear $\mathcal{O}(N)$ (FlashAttention)",
            "Tensor Cores / FlashAttn-2",
            "LLaMA, GPT-4, Mistral, Production"
        ]
    }
    st.table(pd.DataFrame(summary_data))
