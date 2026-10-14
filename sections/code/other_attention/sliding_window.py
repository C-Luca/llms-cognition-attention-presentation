"""
========================================================================================
Module: Attention Mechanisms: Causal vs. Sliding Window (arXiv:2608.28444) vs. FlashAttention
Presentation: Modern LLM Architectures & Memory Systems
Reference: Sliding Window Attention with Attention Sinks (arXiv:2608.28444)
========================================================================================

This module provides an interactive visual and algorithmic walkthrough of:
1. Full Causal Attention vs. SWA with Attention Sinks: SWA(w, s=4) (arXiv:2608.28444)
2. Why Naive Sliding Window Fails: The "Attention Sink" Softmax Phenomenon
3. KV Cache Mechanics: Circular Ring Buffer in GPU VRAM
4. How FlashAttention Works: Hardware-Aware Tiling (SRAM vs. HBM)
5. Step-by-Step PyTorch Implementation
"""

from typing import Dict, Tuple
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server environments
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
import seaborn as sns
import streamlit as st
import torch


# ======================================================================================
# 1. PyTorch Mask Generation Logic
# ======================================================================================

def create_causal_mask(seq_len: int) -> torch.Tensor:
    """
    Generate classic causal attention mask (Vaswani et al., 2017).
    Query token i can only attend to key tokens j <= i, forming a dense lower triangle.
    """
    return torch.tril(torch.ones(seq_len, seq_len, dtype=torch.float32))


def create_naive_sliding_window_mask(seq_len: int, window_size: int) -> torch.Tensor:
    """
    Generate naive sliding window mask without sinks.
    Each token only looks back at maximum w previous tokens: max(0, i - w + 1) <= j <= i.
    
    Warning: Without attention sinks, this policy causes catastrophic Softmax collapse
    once initial tokens pass outside the sliding window.
    """
    causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.float32))
    band_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.float32), diagonal=-(window_size - 1))
    return causal_mask * band_mask


def create_swa_paper_mask(seq_len: int, window_size: int = 64, num_sinks: int = 4) -> torch.Tensor:
    """
    Generate SWA with Attention Sinks as defined in arXiv:2608.28444:
        SWA(w, s) where w is total window budget and s is number of sinks (s=4).
    
    Each query token t attends to:
      1. First s tokens (Attention Sinks): j in [0, s - 1]
      2. (w - s) most recent tokens: j in [max(s, t - (w - s) + 1), t]
    
    Active attention connections per query are capped at maximum w tokens.
    """
    causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.float32))
    
    local_span = max(1, window_size - num_sinks)
    band_mask = torch.triu(
        torch.ones(seq_len, seq_len, dtype=torch.float32), 
        diagonal=-(local_span - 1)
    )
    
    swa_mask = causal_mask * band_mask
    
    # Pin the first s columns (Attention Sinks) active for all rows
    effective_sinks = min(num_sinks, seq_len)
    if effective_sinks > 0:
        swa_mask[:, :effective_sinks] = causal_mask[:, :effective_sinks]
        
    return swa_mask


def create_dilated_sliding_window_mask(seq_len: int, window_size: int, dilation_rate: int = 2) -> torch.Tensor:
    """
    Generate Dilated Sliding Window mask (Longformer, Beltagy et al., 2020).
    Tokens look back across window w, skipping every `dilation_rate` tokens.
    """
    naive_mask = create_naive_sliding_window_mask(seq_len, window_size)
    row_idx = torch.arange(seq_len, dtype=torch.long).unsqueeze(1)
    col_idx = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
    distance = row_idx - col_idx
    dilation_condition = (distance % dilation_rate == 0) & (distance >= 0)
    return naive_mask * dilation_condition.float()


# ======================================================================================
# 2. Structural Metrics & Compute Savings Calculation
# ======================================================================================

def calculate_paper_mask_metrics(
    causal_mask: torch.Tensor,
    naive_mask: torch.Tensor,
    swa_paper_mask: torch.Tensor,
    seq_len: int
) -> Dict[str, Dict[str, float]]:
    """
    Calculates active attention elements, matrix density, and compute savings vs causal baseline.
    """
    total_elements = seq_len * seq_len
    active_causal = int(causal_mask.sum().item())
    active_naive = int(naive_mask.sum().item())
    active_paper = int(swa_paper_mask.sum().item())
    
    savings_naive = (1.0 - (active_naive / active_causal)) * 100.0 if active_causal > 0 else 0.0
    savings_paper = (1.0 - (active_paper / active_causal)) * 100.0 if active_causal > 0 else 0.0
    
    return {
        "causal": {
            "active": active_causal,
            "savings": 0.0,
            "density": (active_causal / total_elements) * 100.0,
            "complexity": "O(N²)",
        },
        "naive": {
            "active": active_naive,
            "savings": savings_naive,
            "density": (active_naive / total_elements) * 100.0,
            "complexity": "O(N × w)",
        },
        "paper_swa": {
            "active": active_paper,
            "savings": savings_paper,
            "density": (active_paper / total_elements) * 100.0,
            "complexity": "O(N × w)",
        },
    }


# ======================================================================================
# 3. Visualizations: Side-by-Side Heatmaps (Matplotlib & Seaborn)
# ======================================================================================

def plot_paper_attention_heatmaps(
    causal_mask: torch.Tensor,
    naive_mask: torch.Tensor,
    swa_paper_mask: torch.Tensor,
    metrics: Dict[str, Dict[str, float]],
    seq_len: int,
    window_size: int,
    num_sinks: int = 4
) -> plt.Figure:
    """
    Renders side-by-side heatmaps illustrating:
    1. Full Causal Attention (dense quadratic triangle)
    2. Naive Sliding Window (pure local band, no sinks)
    3. SWA(w, s) from arXiv:2608.28444 (pinned attention sink columns + local band)
    """
    bg_color = "#0f172a"        # Deep slate (masked element / 0)
    active_color = "#38bdf8"    # Vibrant sky blue (active connection / 1)
    card_bg = "#090d16"         # Outer canvas background
    border_color = "#334155"    # Subtle divider
    title_color = "#f8fafc"     # High-contrast white
    label_color = "#94a3b8"     # Muted slate for axis labels
    
    cmap = ListedColormap([bg_color, active_color])
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), facecolor=card_bg)
    
    panels = [
        {
            "ax": axes[0],
            "mask": causal_mask,
            "title": "1. Full Causal Attention",
            "subtitle": "Vaswani et al. (2017) [Baseline]",
            "active": metrics["causal"]["active"],
            "reduction_str": "Baseline (0% reduction)",
            "highlight_sinks": False,
        },
        {
            "ax": axes[1],
            "mask": naive_mask,
            "title": f"2. Naive SWA (w={window_size})",
            "subtitle": "No Sinks [Softmax Collapse!]",
            "active": metrics["naive"]["active"],
            "reduction_str": f"-{metrics['naive']['savings']:.1f}% compute",
            "highlight_sinks": False,
        },
        {
            "ax": axes[2],
            "mask": swa_paper_mask,
            "title": f"3. SWA({window_size}, {num_sinks}) [arXiv:2608.28444]",
            "subtitle": "2026 Paper [Sinks + Local Band]",
            "active": metrics["paper_swa"]["active"],
            "reduction_str": f"-{metrics['paper_swa']['savings']:.1f}% compute",
            "highlight_sinks": True,
        },
    ]
    
    for panel in panels:
        ax = panel["ax"]
        mask = panel["mask"]
        
        ax.set_facecolor(bg_color)
        sns.heatmap(
            mask.cpu().numpy(),
            cmap=cmap,
            cbar=False,
            ax=ax,
            xticklabels=False,
            yticklabels=False,
            linewidths=0.0,
            rasterized=True if seq_len > 40 else False
        )
        
        ax.set_xlabel("Key Tokens", fontsize=11, fontweight="bold", color=label_color, labelpad=8)
        ax.set_ylabel("Query Tokens", fontsize=11, fontweight="bold", color=label_color, labelpad=8)
        
        for spine in ax.spines.values():
            spine.set_edgecolor(border_color)
            spine.set_linewidth(1.2)
            spine.set_visible(True)
            
        # Highlight attention sinks column in the 3rd plot
        if panel["highlight_sinks"] and num_sinks > 0:
            effective_sinks = min(num_sinks, seq_len)
            sink_rect = patches.Rectangle(
                (0, 0),
                effective_sinks,
                seq_len,
                linewidth=1.8,
                edgecolor="#f59e0b",
                facecolor="#f59e0b",
                alpha=0.22,
                linestyle="--"
            )
            ax.add_patch(sink_rect)
            ax.text(
                effective_sinks / 2, 
                seq_len * 0.92, 
                f"{effective_sinks} Sinks", 
                color="#fbbf24", 
                fontsize=9.5, 
                fontweight="bold", 
                ha="center"
            )
            
        title_text = (
            f"{panel['title']}\n"
            f"{panel['active']:,} active elements | {panel['reduction_str']}"
        )
        ax.set_title(title_text, fontsize=11.5, fontweight="bold", color=title_color, pad=12)
        
    plt.tight_layout()
    return fig


def plot_flashattention_comparison(n_tokens: int = 16, tile_size: int = 4) -> plt.Figure:
    """
    Renders side-by-side conceptual heatmaps comparing Standard Monolithic Attention
    against FlashAttention Hardware-Aware Tiling: loading Tr x Tc tiles into SRAM.
    """
    bg_color = "#0f172a"
    active_color = "#38bdf8"
    card_bg = "#090d16"
    border_color = "#334155"
    title_color = "#f8fafc"
    label_color = "#94a3b8"
    
    cmap = ListedColormap([bg_color, active_color])
    causal_mat = torch.tril(torch.ones(n_tokens, n_tokens)).cpu().numpy()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.8), facecolor=card_bg)
    
    # Plot 1: Standard Attention (Monolithic in HBM)
    ax1.set_facecolor(bg_color)
    sns.heatmap(causal_mat, cmap=cmap, cbar=False, ax=ax1, xticklabels=False, yticklabels=False)
    ax1.set_xlabel("Key Tokens", fontsize=11, fontweight="bold", color=label_color, labelpad=8)
    ax1.set_ylabel("Query Tokens", fontsize=11, fontweight="bold", color=label_color, labelpad=8)
    ax1.set_title("Standard Attention (Monolithic)\nMaterializes Full N × N Matrix in Slow HBM", fontsize=11.5, fontweight="bold", color=title_color, pad=12)
    
    ax1.text(
        n_tokens / 2, n_tokens / 2,
        "Monolithic N × N Matrix\n(Repeated Reads/Writes to Slow HBM)\nMemory Bandwidth Bound: O(N²)",
        fontsize=10.0, fontweight="bold", color="#f8fafc", ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#1e293b", edgecolor="#ef4444", lw=1.8, alpha=0.92)
    )
    for spine in ax1.spines.values():
        spine.set_edgecolor(border_color)
        spine.set_linewidth(1.2)
        spine.set_visible(True)
        
    # Plot 2: FlashAttention (Tiling in SRAM)
    ax2.set_facecolor(bg_color)
    sns.heatmap(causal_mat, cmap=cmap, cbar=False, ax=ax2, xticklabels=False, yticklabels=False)
    ax2.set_xlabel("Key Tokens", fontsize=11, fontweight="bold", color=label_color, labelpad=8)
    ax2.set_ylabel("Query Tokens", fontsize=11, fontweight="bold", color=label_color, labelpad=8)
    ax2.set_title("FlashAttention (Hardware-Aware Tiling)\nLoads Tr × Tc Tiles Directly into Fast On-Chip SRAM", fontsize=11.5, fontweight="bold", color=title_color, pad=12)
    
    # Draw bold tile grid lines
    for i in range(0, n_tokens + 1, tile_size):
        ax2.axhline(i, color="#f8fafc", linewidth=2.0, alpha=0.85, linestyle="--")
        ax2.axvline(i, color="#f8fafc", linewidth=2.0, alpha=0.85, linestyle="--")
        
    # Highlight active tile in SRAM
    tile_r = 8
    tile_c = 4
    active_patch = patches.Rectangle(
        (tile_c, tile_r), tile_size, tile_size,
        linewidth=2.5, edgecolor="#f59e0b", facecolor="#f59e0b", alpha=0.35
    )
    ax2.add_patch(active_patch)
    
    ax2.annotate(
        "Active Tile in Fast SRAM (19 TB/s)\n• Online Softmax rescale\n• Bypasses Slow HBM (~2 TB/s)",
        xy=(tile_c + tile_size / 2, tile_r + tile_size / 2),
        xytext=(tile_c + tile_size + 1.0, tile_r - 2.2),
        arrowprops=dict(facecolor="#f59e0b", edgecolor="#f59e0b", shrink=0.06, width=2.0, headwidth=7),
        fontsize=9.5, fontweight="bold", color="#fbbf24",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#1e293b", edgecolor="#f59e0b", lw=1.5, alpha=0.95)
    )
    for spine in ax2.spines.values():
        spine.set_edgecolor(border_color)
        spine.set_linewidth(1.2)
        spine.set_visible(True)
        
    plt.tight_layout()
    return fig


# Backwards-compatibility aliases
create_sliding_window_mask = create_swa_paper_mask
create_sliding_window_with_sinks_mask = create_swa_paper_mask
calculate_mask_metrics = calculate_paper_mask_metrics
plot_attention_heatmaps = plot_paper_attention_heatmaps


# ======================================================================================
# 4. Streamlit UI Rendering Entrypoint
# ======================================================================================

def render_sliding_window():
    """
    Primary rendering function for the Attention Mechanisms presentation module.
    Explains EXCLUSIVELY HOW THE MECHANISMS WORK: mathematical intuition, memory mechanics,
    and hardware tiling, without benchmark result tables.
    """
    # ----------------------------------------------------------------------------------
    # Header & Core Architectural Overview
    # ----------------------------------------------------------------------------------
    st.header("How Attention Works: Causal vs. Sliding Window (arXiv:2608.28444) vs. FlashAttention")
    st.caption("A Mechanics-First Deep Dive into Memory Footprints, Attention Sinks, and GPU Tiling")
    
    st.markdown(r"""
    In modern Large Language Models (LLMs), self-attention is governed by the core equation:
    $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^\top}{\sqrt{d}}\right) V$$
    
    During autoregressive token generation, storing past Keys ($K$) and Values ($V$) creates the **KV Cache**.
    As the sequence length grows, the KV Cache expands linearly $\mathcal{O}(N)$ in memory, and attention operations 
    scale quadratically $\mathcal{O}(N^2)$.
    
    This module breaks down the mathematical mechanisms and hardware architectures developed to solve this bottleneck.
    """)
    
    st.divider()
    
    # ----------------------------------------------------------------------------------
    # Section 1: Attention Masks Interactive Demonstration
    # ----------------------------------------------------------------------------------
    st.subheader("1. How Attention Masks Work: Full Causal vs. Naive SWA vs. SWA with Sinks")
    st.markdown("""
    The attention mask determines which historical tokens the current query token is permitted to attend to.
    Adjust the sliders below to see how the geometric attention pattern transforms:
    """)
    
    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns(3)
    with ctrl_col1:
        seq_len = st.slider(
            "Sequence Length (N)",
            min_value=10,
            max_value=100,
            value=24,
            step=1,
            help="Total number of tokens in the attention matrix (dimension N x N)."
        )
    with ctrl_col2:
        window_size = st.slider(
            "Window Size (w)",
            min_value=4,
            max_value=50,
            value=8,
            step=1,
            help="Total active token budget each query attends to."
        )
    with ctrl_col3:
        num_sinks = st.slider(
            "Attention Sinks (s)",
            min_value=1,
            max_value=min(8, window_size),
            value=min(4, window_size),
            step=1,
            help="Number of initial tokens pinned permanently (fixed at s=4 in arXiv:2608.28444)."
        )
        
    causal_mask = create_causal_mask(seq_len)
    naive_mask = create_naive_sliding_window_mask(seq_len, window_size)
    paper_mask = create_swa_paper_mask(seq_len, window_size, num_sinks)
    
    metrics = calculate_paper_mask_metrics(causal_mask, naive_mask, paper_mask, seq_len)
    
    # Structural metric cards
    m_col1, m_col2, m_col3 = st.columns(3)
    with m_col1:
        st.metric(
            label="1. Full Causal Attention",
            value=f"{metrics['causal']['active']:,} active elements",
            delta=f"{metrics['causal']['density']:.1f}% density",
            delta_color="off",
            help="Dense lower-triangular matrix. Each token attends to all prior tokens."
        )
        st.caption(f"**Compute Complexity:** $\\mathcal{{O}}(N^2) = \\mathcal{{O}}({seq_len}^2)$")
        
    with m_col2:
        st.metric(
            label=f"2. Naive SWA (w={window_size})",
            value=f"{metrics['naive']['active']:,} active elements",
            delta=f"-{metrics['naive']['savings']:.1f}% compute vs Full",
            delta_color="inverse",
            help="Pure local band without sinks. Attention Softmax collapses once initial tokens pass out of window."
        )
        st.caption("**Failure Mode:** Softmax collapse without sinks")
        
    with m_col3:
        st.metric(
            label=f"3. SWA({window_size}, {num_sinks}) [arXiv:2608.28444]",
            value=f"{metrics['paper_swa']['active']:,} active elements",
            delta=f"-{metrics['paper_swa']['savings']:.1f}% compute vs Full",
            delta_color="inverse",
            help="SWA with sinks: pins the first s columns + slides local window w-s. 100% stable infinite generation."
        )
        st.caption(f"**Complexity:** $\\mathcal{{O}}(N \\times w)$ with complete numerical stability")

    # Heatmaps side-by-side
    fig_heatmaps = plot_paper_attention_heatmaps(
        causal_mask=causal_mask,
        naive_mask=naive_mask,
        swa_paper_mask=paper_mask,
        metrics=metrics,
        seq_len=seq_len,
        window_size=window_size,
        num_sinks=num_sinks,
    )
    st.pyplot(fig_heatmaps, use_container_width=True)
    plt.close(fig_heatmaps)
    
    st.markdown(f"""
    > **Visual Observations:**
    > - **Plot 1 (Full Causal):** Dense lower triangle. The computational surface area expands quadratically as $N$ grows.
    > - **Plot 2 (Naive SWA):** Banded diagonal of width $w$. When generating beyond token $w$, the model completely loses contact with the start of the prompt.
    > - **Plot 3 (SWA with Sinks):** Notice the **golden vertical strip** on the left (first $s={num_sinks}$ columns). 
    >   These sink tokens remain persistently visible to every future query, while the remainder of the attention budget glides locally.
    """)
    
    st.divider()

    # ----------------------------------------------------------------------------------
    # Section 2: Why Naive SWA Fails & What Attention Sinks Are
    # ----------------------------------------------------------------------------------
    st.subheader("2. Why Naive SWA Fails: The 'Attention Sink' Phenomenon")
    
    sink_col1, sink_col2 = st.columns([1, 1])
    
    with sink_col1:
        st.markdown("#### The Mathematical Constraint of Softmax")
        st.markdown(r"""
        For any query token $q_i$, attention weights are normalized across key positions $j$:
        $$\alpha_{ij} = \frac{\exp(q_i k_j^\top / \sqrt{d})}{\sum_{m} \exp(q_i k_m^\top / \sqrt{d})}, \quad \sum_{j} \alpha_{ij} = 1.0$$
        
        What happens when a query token $q_i$ **does not need** context information (for example, generating punctuation or common syntactic filler)?
        - The model is **mathematically forced** by the Softmax normalizer to distribute a total probability mass equal to $1.0$.
        - During pre-training, Transformers naturally discover a clever mechanism: **they dump excess, unnecessary attention scores onto the initial token** in the sequence (the `<s>` / BOS token).
        - The initial token acts as an **Attention Sink**: a repository absorbing residual Softmax probability mass across all layers and heads.
        """)
        
    with sink_col2:
        st.markdown("#### The Collapse of Naive Sliding Windows")
        st.markdown(r"""
        If we implement a naive sliding window (evicting tokens strictly based on age):
        1. As long as the `<s>` token resides within the lookback window ($t < w$), text generation proceeds normally.
        2. The moment token generation steps beyond $w$, **the BOS token is evicted from the KV Cache**!
        3. The query still attempts to dump surplus probability mass, but the sink token is gone.
        4. Result: The Softmax mass is forced to arbitrarily redistribute across nearby local tokens, triggering **catastrophic distribution collapse** (perplexity explodes to infinity).
        
        **The arXiv:2608.28444 Solution:**
        Retaining just the **first 4 tokens** ($s=4$) as permanent anchors in the KV Cache stabilizes the Softmax distribution permanently, enabling infinite-length generation without fine-tuning!
        """)

    st.markdown("```\nToken Stream: [Sink 0, Sink 1, Sink 2, Sink 3] ... [Token t-3, Token t-2, Token t-1, Token t (Local Window)]\n```")
    st.caption("Figure: Memory layout for SWA(w, s=4) combining persistent attention sinks with a circular rolling local window.")

    st.divider()

    # ----------------------------------------------------------------------------------
    # Section 3: KV Cache Mechanics & Circular Ring Buffer
    # ----------------------------------------------------------------------------------
    st.subheader("3. KV Cache Mechanics: The Circular Ring Buffer")
    
    mem_left, mem_right = st.columns([1, 1])
    
    with mem_left:
        st.markdown("#### KV Cache Memory Sizing Formula")
        st.markdown(r"""
        During autoregressive decoding, GPUs retain Key and Value vectors for every layer and attention head:
        $$\text{KV Cache (Bytes)} = 2 \times L \times H \times d \times N \times \text{BytesPerElement}$$
        Where:
        - $2$: separate Key ($K$) and Value ($V$) tensors
        - $L$: number of Transformer layers (e.g., $32$)
        - $H$: number of attention heads (e.g., $32$)
        - $d$: head dimension (e.g., $128$)
        - $N$: context length in tokens
        - $\text{BytesPerElement}$: $2$ bytes for FP16 / BF16
        """)
        
        c_seq = st.select_slider(
            "Select Context Length (N tokens):",
            options=[2048, 4096, 8192, 16384, 32768, 65536, 131072],
            value=32768
        )
        
        l_layers, h_heads, d_dim, b_bytes = 32, 32, 128, 2
        full_kv_gb = (2 * l_layers * h_heads * d_dim * c_seq * b_bytes) / (1024**3)
        
        w_budget = 1024
        swa_kv_gb = (2 * l_layers * h_heads * d_dim * min(c_seq, w_budget) * b_bytes) / (1024**3)
        
        kv_c1, kv_c2 = st.columns(2)
        kv_c1.metric("Full Attention KV Cache", f"{full_kv_gb:.2f} GB VRAM")
        kv_c2.metric(f"SWA (Window {w_budget})", f"{swa_kv_gb:.2f} GB VRAM", delta=f"-{(1.0 - swa_kv_gb/full_kv_gb)*100.0:.1f}% VRAM")
        
    with mem_right:
        st.markdown("#### Circular Rolling Buffer in VRAM")
        st.markdown(r"""
        Instead of dynamically allocating memory up to 128k tokens:
        - SWA pre-allocates a fixed block of exactly $w$ token slots in GPU memory.
        - **Slots 0..3:** Permanently reserved for the 4 sink tokens and **never overwritten**.
        - **Slots 4..w-1:** Form a circular ring buffer. New token $t$ overwrites the slot:
          $$\text{slot} = 4 + ((t - 4) \bmod (w - 4))$$
        - Result: GPU VRAM consumption is **completely flat and constant**, whether decoding 100 or 1,000,000 tokens!
        """)

    st.divider()

    # ----------------------------------------------------------------------------------
    # Section 4: FlashAttention: Hardware-Aware Tiling
    # ----------------------------------------------------------------------------------
    st.subheader("4. FlashAttention: Hardware-Aware Tiling (HBM vs SRAM)")
    
    st.markdown(
        "FlashAttention does not change the math (exact same output as standard attention). "
        "Instead, it uses **Tiling** and **Recomputation** to fuse the kernels, "
        "keeping intermediate $Q \\times K^T$ scores in ultra-fast SRAM and "
        "avoiding expensive reads/writes to slow HBM."
    )
    
    st.markdown("#### Hardware Complexity Comparison (N = 32,768, d = 128)")
    
    fa_n = 32768
    fa_d = 128
    
    total_flops = 4 * (fa_n ** 2) * fa_d
    flops_gflops = total_flops / 1e9
    
    std_hbm_bytes = (4 * (fa_n ** 2) * 2) + (4 * fa_n * fa_d * 2)
    std_hbm_gb = std_hbm_bytes / (1024 ** 3)
    
    fa_hbm_bytes = 4 * fa_n * fa_d * 2
    fa_hbm_mb = fa_hbm_bytes / (1024 ** 2)
    hbm_reduction_pct = (1.0 - (fa_hbm_bytes / std_hbm_bytes)) * 100.0
    
    col_fa1, col_fa2, col_fa3, col_fa4 = st.columns(4)
    with col_fa1:
        st.metric(
            label="1. Math Operations (FLOPs)",
            value=f"{flops_gflops:.1f} GFLOPs",
            delta="0% (Identical Math)",
            delta_color="off",
            help="Formula: 4 * N² * d. Both Standard and FlashAttention compute identical matrix operations."
        )
        st.caption("Arithmetic: $\\mathcal{O}(N^2 d)$")
        
    with col_fa2:
        st.metric(
            label="2. Standard Attention HBM IO",
            value=f"{std_hbm_gb:.2f} GB / head",
            delta="O(N²) High Latency IO",
            delta_color="off",
            help="Writes and reads the 2 GB N x N matrix 4 times through high-latency off-chip HBM."
        )
        st.caption("Memory IO: $\\mathcal{O}(N^2)$ (Memory Bound)")
        
    with col_fa3:
        st.metric(
            label="3. FlashAttention HBM IO",
            value=f"{fa_hbm_mb:.1f} MB / head",
            delta=f"-{hbm_reduction_pct:.1f}% HBM Traffic",
            delta_color="inverse",
            help="Intermediate tiles stay strictly in SRAM. The N x N matrix is never written to HBM."
        )
        st.caption("Memory IO: $\\mathcal{O}(N \\times d)$ (Compute Bound)")
        
    with col_fa4:
        st.metric(
            label="4. Memory Bandwidth (NVIDIA A100)",
            value="19.0 TB/s (SRAM)",
            delta="~10x vs HBM (2.0 TB/s)",
            delta_color="normal",
            help="On-chip SRAM provides an order of magnitude faster memory bandwidth than off-chip HBM."
        )
        st.caption("Speedup: 2x - 4x Wall-Clock")

    # Side-by-side conceptual architecture plot
    st.markdown("#### Conceptual Architecture: Monolithic HBM vs. Tiled SRAM Execution")
    fig_fa = plot_flashattention_comparison(n_tokens=16, tile_size=4)
    st.pyplot(fig_fa, use_container_width=True)
    plt.close(fig_fa)
    
    fa_detail_left, fa_detail_right = st.columns(2)
    with fa_detail_left:
        st.markdown(r"""
        **Pillar 1: Tiling & Online Softmax**
        - Standard Softmax requires scaling by $\sum_{j=1}^N \exp(S_{ij})$, forcing all $N$ columns to be generated before dividing.
        - FlashAttention splits $Q, K, V$ into blocks $T_r \times T_c$ that fit in **fast SRAM** (~100–228 KB per Streaming Multiprocessor).
        - It uses **Online Softmax**: tracking running row-maxima $m$ and normalizer sums $\ell$, iteratively rescaling the accumulator tile-by-tile without materializing the $N \times N$ matrix!
        """)
    with fa_detail_right:
        st.markdown(r"""
        **Pillar 2: Kernel Fusion & Recomputation**
        - In traditional training, the $N \times N$ attention matrix is saved to HBM during forward pass for backward gradient computation.
        - FlashAttention **never stores the $N \times N$ matrix**.
        - In the backward pass, it quickly **recomputes** the $T_r \times T_c$ tiles on-the-fly in SRAM from stored $Q, K, V$. Because arithmetic on GPU is fast while memory transfer is slow, this recomputation speeds up wall-clock execution and frees massive VRAM!
        """)

    st.divider()

    # ----------------------------------------------------------------------------------
    # Section 5: Step-by-Step PyTorch Implementation
    # ----------------------------------------------------------------------------------
    st.subheader("5. Step-by-Step PyTorch Implementation")
    st.markdown("Clean, vectorized PyTorch code implementing SWA with Attention Sinks and hardware-dispatched FlashAttention:")
    
    st.code(r'''import torch

def create_swa_paper_mask(seq_len: int, window_size: int = 64, num_sinks: int = 4) -> torch.Tensor:
    """
    Generate Sliding Window Attention Mask with Attention Sinks (arXiv:2608.28444).
    Each token t attends to:
      - First s sink tokens: [0, s-1]
      - (w - s) local tokens: [max(s, t - (w - s) + 1), t]
    """
    # 1. Lower-triangular causal condition (j <= i)
    causal = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.float32))
    
    # 2. Local banded window of width (window_size - num_sinks)
    local_span = max(1, window_size - num_sinks)
    band = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.float32), diagonal=-(local_span - 1))
    swa_mask = causal * band
    
    # 3. Pin first s attention sink columns active across all rows
    effective_sinks = min(num_sinks, seq_len)
    swa_mask[:, :effective_sinks] = causal[:, :effective_sinks]
    
    return swa_mask  # Shape: [seq_len, seq_len]

# Hardware-Aware Execution via PyTorch 2.0+ FlashAttention:
def run_flash_attention(query, key, value, is_causal=True):
    """Executes fused FlashAttention kernel with O(N * d) memory bandwidth IO."""
    return torch.nn.functional.scaled_dot_product_attention(
        query, key, value, is_causal=is_causal
    )
''', language="python")


# ======================================================================================
# Standalone Execution Entrypoint
# ======================================================================================

if __name__ == "__main__":
    st.set_page_config(
        page_title="Attention Mechanisms: Causal vs. SWA (arXiv:2608.28444) vs. FlashAttention",
        page_icon=None,
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    render_sliding_window()

