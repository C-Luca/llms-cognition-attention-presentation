"""
========================================================================================
Module: Sliding Window Attention vs. Classic Causal Attention
Presentation: Attention Mechanisms & Modern LLM Architectures (arXiv:2608.28444)
Authors: University Presentation Team
========================================================================================

This module provides an interactive visual comparison between:
1. Classic Causal Attention ("Attention Is All You Need", Vaswani et al., 2017)
2. Sliding Window Attention ("Sliding-window beats linear attention", arXiv:2608.28444)
3. Dilated Sliding Window Attention (Longformer, Beltagy et al., 2020)

It demonstrates PyTorch mask construction, active computational elements,
memory footprint reductions, and theoretical complexity trade-offs.
"""

from typing import Dict, Tuple
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend suitable for server environments
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
import streamlit as st
import torch


# ======================================================================================
# 1. PyTorch Mask Generation Logic
# ======================================================================================

def create_causal_mask(seq_len: int) -> torch.Tensor:
    """
    Generate Classic Causal Attention Mask (Vaswani et al., 2017).
    
    Constructs a lower-triangular matrix of ones of size (N, N).
    In causal (autoregressive) generation, query token i can only attend
    to key tokens j <= i, strictly prohibiting access to future tokens.
    
    Args:
        seq_len (int): Total sequence length N.
        
    Returns:
        torch.Tensor: Binary float mask of shape (N, N) where 1 indicates an active
                      attention connection and 0 indicates a masked connection.
    """
    # torch.tril returns the lower triangular part of the matrix (including diagonal)
    return torch.tril(torch.ones(seq_len, seq_len, dtype=torch.float32))


def create_sliding_window_mask(seq_len: int, window_size: int) -> torch.Tensor:
    """
    Generate Sliding Window Attention Mask (arXiv:2608.28444, Child et al., 2019).
    
    Constructs a banded lower-triangular matrix of size (N, N).
    Each query token i attends to itself and at most w preceding key tokens:
        max(0, i - window_size) <= j <= i
    
    Args:
        seq_len (int): Total sequence length N.
        window_size (int): Lookback window size w.
        
    Returns:
        torch.Tensor: Banded causal mask of shape (N, N) with active local elements.
    """
    # Enforce causality: key index j <= query index i
    causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.float32))
    
    # Enforce window boundary: key index j >= i - window_size
    # diagonal=-window_size keeps elements on and above the subdiagonal at -window_size
    band_mask = torch.triu(
        torch.ones(seq_len, seq_len, dtype=torch.float32), 
        diagonal=-window_size
    )
    
    # Element-wise intersection yields the causal banded window
    return causal_mask * band_mask


def create_dilated_sliding_window_mask(
    seq_len: int, 
    window_size: int, 
    dilation_rate: int = 2
) -> torch.Tensor:
    """
    Generate Dilated Sliding Window Attention Mask (Beltagy et al., 2020).
    
    Tokens look back across the window w, but skip every `dilation_rate` tokens:
        (i - j) % dilation_rate == 0  and  0 <= i - j <= w
    This expands the effective receptive field coverage while sparsifying
    the computation even further.
    
    Args:
        seq_len (int): Total sequence length N.
        window_size (int): Lookback window size w.
        dilation_rate (int): Stride rate (default: 2, skips every 2nd token).
        
    Returns:
        torch.Tensor: Dilated banded causal mask of shape (N, N).
    """
    # Base sliding window mask
    sliding_mask = create_sliding_window_mask(seq_len, window_size)
    
    # Construct relative distance grid: distance[i, j] = i - j
    row_idx = torch.arange(seq_len, dtype=torch.long).unsqueeze(1)
    col_idx = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
    distance = row_idx - col_idx
    
    # Keep only positions where distance is non-negative and a multiple of dilation_rate
    dilation_condition = (distance % dilation_rate == 0) & (distance >= 0)
    
    return sliding_mask * dilation_condition.float()


def create_sliding_window_with_sinks_mask(
    seq_len: int, 
    window_size: int, 
    num_sinks: int = 4
) -> torch.Tensor:
    """
    Generate Sliding Window Attention Mask with Initial Attention Sinks
    (StreamingLLM / Xiao et al., 2023 / arXiv:2608.28444).
    
    In addition to attending to the local window of w tokens, query tokens
    maintain attention to the first `num_sinks` tokens of the sequence (e.g. BOS).
    This prevents softmax distribution collapse and stabilizes long generation.
    
    Args:
        seq_len (int): Sequence length N.
        window_size (int): Lookback window size w.
        num_sinks (int): Number of initial sink tokens to preserve.
        
    Returns:
        torch.Tensor: Attention mask of shape (N, N) with sinks and sliding window.
    """
    base_swa = create_sliding_window_mask(seq_len, window_size)
    causal = create_causal_mask(seq_len)
    
    effective_sinks = min(num_sinks, seq_len)
    if effective_sinks > 0:
        # Preserve initial columns for all rows while respecting causality
        base_swa[:, :effective_sinks] = causal[:, :effective_sinks]
        
    return base_swa


# ======================================================================================
# 2. Computational Statistics & Metrics Calculation
# ======================================================================================

def calculate_mask_metrics(
    causal_mask: torch.Tensor,
    sliding_mask: torch.Tensor,
    dilated_mask: torch.Tensor,
    seq_len: int
) -> Dict[str, Dict[str, float]]:
    """
    Calculate active attention elements, compute savings, and matrix density.
    
    Args:
        causal_mask: Classic causal mask tensor.
        sliding_mask: Sliding window mask tensor.
        dilated_mask: Dilated sliding window mask tensor.
        seq_len: Sequence length N.
        
    Returns:
        dict: Statistical metrics for each attention variant.
    """
    total_elements = seq_len * seq_len
    
    active_causal = int(causal_mask.sum().item())
    active_sliding = int(sliding_mask.sum().item())
    active_dilated = int(dilated_mask.sum().item())
    
    # Compute reduction relative to classic causal baseline
    savings_sliding = (
        (1.0 - (active_sliding / active_causal)) * 100.0 if active_causal > 0 else 0.0
    )
    savings_dilated = (
        (1.0 - (active_dilated / active_causal)) * 100.0 if active_causal > 0 else 0.0
    )
    
    # Matrix density (percentage of N x N that is non-zero)
    density_causal = (active_causal / total_elements) * 100.0 if total_elements > 0 else 0.0
    density_sliding = (active_sliding / total_elements) * 100.0 if total_elements > 0 else 0.0
    density_dilated = (active_dilated / total_elements) * 100.0 if total_elements > 0 else 0.0
    
    return {
        "causal": {
            "active": active_causal,
            "savings": 0.0,
            "density": density_causal,
            "complexity": "O(N²)",
        },
        "sliding": {
            "active": active_sliding,
            "savings": savings_sliding,
            "density": density_sliding,
            "complexity": "O(N × w)",
        },
        "dilated": {
            "active": active_dilated,
            "savings": savings_dilated,
            "density": density_dilated,
            "complexity": "O(N × ⌈w/2⌉)",
        },
    }


# ======================================================================================
# 3. Visualization: Side-by-Side Heatmaps (Matplotlib & Seaborn)
# ======================================================================================

def plot_attention_heatmaps(
    causal_mask: torch.Tensor,
    sliding_mask: torch.Tensor,
    dilated_mask: torch.Tensor,
    metrics: Dict[str, Dict[str, float]],
    seq_len: int,
    window_size: int,
) -> plt.Figure:
    """
    Renders side-by-side attention mask heatmaps styled to match the presentation theme.
    
    Color palette:
      - Inactive / Masked: Deep Slate (#0f172a)
      - Active / Attending: Vibrant Sky Blue (#38bdf8)
      - Card Background: Dark Navy (#0b1120)
      - Borders: Slate Gray (#334155)
      
    Args:
        causal_mask: Classic causal mask tensor.
        sliding_mask: Sliding window mask tensor.
        dilated_mask: Dilated sliding window mask tensor.
        metrics: Dictionary containing active element counts and savings.
        seq_len: Sequence length N.
        window_size: Window size w.
        
    Returns:
        plt.Figure: Styled Matplotlib figure with 3 side-by-side heatmaps.
    """
    # Color palette matching the presentation theme
    bg_color = "#0f172a"        # Deep slate (masked 0)
    active_color = "#38bdf8"    # Vibrant sky blue (active 1)
    card_bg = "#090d16"         # Outer figure canvas
    border_color = "#334155"    # Subtle divider
    title_color = "#f8fafc"     # High-contrast white
    label_color = "#94a3b8"     # Muted slate for axis labels
    
    cmap = ListedColormap([bg_color, active_color])
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.3), facecolor=card_bg)
    
    panels = [
        {
            "ax": axes[0],
            "mask": causal_mask,
            "title": "Classic Causal Attention",
            "subtitle": "Vaswani et al. (2017)",
            "stats": metrics["causal"],
            "reduction_str": "Baseline (0% reduction)",
        },
        {
            "ax": axes[1],
            "mask": sliding_mask,
            "title": f"Sliding Window Attention (w={window_size})",
            "subtitle": "arXiv:2608.28444",
            "stats": metrics["sliding"],
            "reduction_str": f"-{metrics['sliding']['savings']:.1f}% compute reduction",
        },
        {
            "ax": axes[2],
            "mask": dilated_mask,
            "title": f"Dilated Sliding Window (w={window_size})",
            "subtitle": "Beltagy et al. (2020) [stride=2]",
            "stats": metrics["dilated"],
            "reduction_str": f"-{metrics['dilated']['savings']:.1f}% compute reduction",
        },
    ]
    
    for panel in panels:
        ax = panel["ax"]
        mask = panel["mask"]
        stats = panel["stats"]
        
        ax.set_facecolor(bg_color)
        
        # Seaborn heatmap rendering
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
        
        # Keep axis labels, remove tick marks
        ax.set_xlabel("Key Tokens", fontsize=11, fontweight="bold", color=label_color, labelpad=8)
        ax.set_ylabel("Query Tokens", fontsize=11, fontweight="bold", color=label_color, labelpad=8)
        
        # Clean subtle borders
        for spine in ax.spines.values():
            spine.set_edgecolor(border_color)
            spine.set_linewidth(1.2)
            spine.set_visible(True)
            
        # Descriptive subplot title displaying active element count and savings
        title_text = (
            f"{panel['title']}\n"
            f"{stats['active']:,} active elements | {panel['reduction_str']}"
        )
        ax.set_title(title_text, fontsize=11.5, fontweight="bold", color=title_color, pad=12)
        
    plt.tight_layout()
    return fig


# ======================================================================================
# 4. Streamlit UI Rendering Entrypoint
# ======================================================================================

def render_sliding_window():
    """
    Primary rendering function for the Sliding Window Attention module.
    Can be imported by multi-page presentation apps or executed standalone.
    """
    # Header & Academic Framing
    st.header("⚡ Sliding Window Attention vs. Classic Causal Attention")
    st.caption("Comparing 'Attention Is All You Need' (Vaswani et al., 2017) vs. 'Sliding-window beats linear attention' (arXiv:2608.28444)")
    
    st.markdown(r"""
    Modern LLMs traditionally rely on **Full Causal Self-Attention**, where each token attends to all prior tokens.
    While expressive, this requires $\mathcal{O}(N^2)$ compute and linear $\mathcal{O}(N)$ memory per layer for the KV Cache.
    
    Recent research (**arXiv:2608.28444**) demonstrates that **Sliding Window Attention (SWA)** outperforms recurrent / linear-attention 
    approaches by retaining precise softmax focus over the most recent local window $w$, bounding compute to $\mathcal{O}(N \times w)$ 
    while preserving language modeling perplexity.
    """)
    
    st.divider()
    
    # ----------------------------------------------------------------------------------
    # Controls Section
    # ----------------------------------------------------------------------------------
    st.subheader("🎛️ Interactive Mask Controls")
    
    ctrl_col1, ctrl_col2 = st.columns(2)
    with ctrl_col1:
        seq_len = st.slider(
            "Sequence Length (N)",
            min_value=10,
            max_value=100,
            value=20,
            step=1,
            help="Total number of tokens in the sequence (matrix dimension N x N)."
        )
    with ctrl_col2:
        window_size = st.slider(
            "Window Size (w)",
            min_value=1,
            max_value=50,
            value=4,
            step=1,
            help="Maximum number of preceding tokens each token looks back to attend to."
        )
        
    if window_size >= seq_len:
        st.info(
            f"💡 **Note:** Window size ($w={window_size}$) is greater than or equal to sequence length ($N={seq_len}$). "
            "Sliding Window Attention covers the entire past context, becoming identical to Classic Causal Attention."
        )
        
    # ----------------------------------------------------------------------------------
    # PyTorch Mask Computation
    # ----------------------------------------------------------------------------------
    causal_mask = create_causal_mask(seq_len)
    sliding_mask = create_sliding_window_mask(seq_len, window_size)
    dilated_mask = create_dilated_sliding_window_mask(seq_len, window_size, dilation_rate=2)
    
    metrics = calculate_mask_metrics(causal_mask, sliding_mask, dilated_mask, seq_len)
    
    # ----------------------------------------------------------------------------------
    # Metric Summary Cards
    # ----------------------------------------------------------------------------------
    st.markdown("#### 📊 Compute & Active Element Statistics")
    m_col1, m_col2, m_col3 = st.columns(3)
    
    with m_col1:
        st.metric(
            label="1. Classic Causal Attention",
            value=f"{metrics['causal']['active']:,} active",
            delta=f"{metrics['causal']['density']:.1f}% matrix density",
            delta_color="off",
            help="Dense lower-triangular matrix. Each token attends to all prior tokens."
        )
        st.caption(f"**Complexity:** $\\mathcal{{O}}(N^2) = \\mathcal{{O}}({seq_len}^2)$")
        
    with m_col2:
        st.metric(
            label=f"2. Sliding Window Attention (w={window_size})",
            value=f"{metrics['sliding']['active']:,} active",
            delta=f"-{metrics['sliding']['savings']:.1f}% compute vs Causal",
            delta_color="inverse",
            help="Banded attention matrix. Each token attends to at most w preceding tokens."
        )
        st.caption(f"**Complexity:** $\\mathcal{{O}}(N \\times w) = \\mathcal{{O}}({seq_len} \\times {window_size})$")
        
    with m_col3:
        st.metric(
            label=f"3. Dilated Sliding Window (w={window_size})",
            value=f"{metrics['dilated']['active']:,} active",
            delta=f"-{metrics['dilated']['savings']:.1f}% compute vs Causal",
            delta_color="inverse",
            help="Dilated banded matrix with stride=2 (skips every 2nd token)."
        )
        st.caption(f"**Complexity:** $\\mathcal{{O}}(N \\times \\lceil w/2 \\rceil)$")

    # ----------------------------------------------------------------------------------
    # Visualizations: Heatmaps Side-by-Side
    # ----------------------------------------------------------------------------------
    st.markdown("#### 🗺️ Attention Masks Comparison Heatmaps")
    st.markdown(
        "*Dark slate indicates masked/ignored token interactions ($0$); sky blue indicates active attention weights ($1$). "
        "Rows represent **Query Tokens**; columns represent **Key Tokens**.*"
    )
    
    fig = plot_attention_heatmaps(
        causal_mask=causal_mask,
        sliding_mask=sliding_mask,
        dilated_mask=dilated_mask,
        metrics=metrics,
        seq_len=seq_len,
        window_size=window_size,
    )
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    
    # ----------------------------------------------------------------------------------
    # Architectural Breakdown & Analytical Comparison
    # ----------------------------------------------------------------------------------
    st.divider()
    st.subheader("🔍 Architectural Analysis & Theoretical Trade-offs")
    
    a_col1, a_col2, a_col3 = st.columns(3)
    
    with a_col1:
        st.markdown("##### 1. Classic Causal Attention")
        st.markdown(r"""
        - **Mechanism:** Lower-triangular matrix: $A_{ij} = 1$ for all $j \le i$.
        - **Compute:** $\frac{N(N+1)}{2}$ elements $\implies \mathcal{O}(N^2)$.
        - **Memory:** KV Cache grows linearly with total sequence length $N$.
        - **Advantage:** Full global context visibility across arbitrary sequence lengths.
        - **Bottleneck:** Prohibitive quadratic cost for contexts $> 32\text{k}$ tokens.
        """)
        
    with a_col2:
        st.markdown(f"##### 2. Sliding Window (w={window_size})")
        st.markdown(rf"""
        - **Mechanism:** Banded matrix: $A_{{ij}} = 1$ for $\max(0, i-w) \le j \le i$.
        - **Compute:** At most $(w+1)$ keys per query $\implies \mathcal{{O}}(N \times w)$.
        - **Memory:** Fixed-size cyclic KV Cache of $w$ tokens per layer.
        - **arXiv:2608.28444 Finding:** Preserves exact softmax attention over local context, outperforming linear-state recurrent approximations.
        - **Architecture:** Core mechanism of Mistral 7B & modern hybrid LLMs.
        """)
        
    with a_col3:
        st.markdown(f"##### 3. Dilated Sliding Window")
        st.markdown(rf"""
        - **Mechanism:** Stride-2 banded matrix: $(i - j) \bmod 2 = 0$ within window $w$.
        - **Compute:** Halves active tokens within the window $\implies \mathcal{{O}}(N \times \lceil w/2 \rceil)$.
        - **Receptive Field:** Reaches up to $w$ tokens in context while saving **{metrics['dilated']['savings']:.1f}%** compute.
        - **Usage:** Multi-scale architectures (Longformer) where alternate layers use dilated vs. dense local windows.
        """)

    # ----------------------------------------------------------------------------------
    # The Attention Sink Phenomenon & KV Cache Memory Impact
    # ----------------------------------------------------------------------------------
    st.divider()
    st.subheader("🧮 Practical Impact: KV Cache Memory & Attention Sinks")
    
    left_impact, right_impact = st.columns([1, 1])
    
    with left_impact:
        st.markdown("#### Real-World KV Cache Memory Explosion")
        st.markdown(r"""
        During autoregressive inference, GPUs store historical Key ($K$) and Value ($V$) tensors in VRAM:
        $$\text{KV Cache Size} = 2 \times L \times H \times D \times N \times \text{Bytes}$$
        For large contexts ($N = 32\text{k}$ to $128\text{k}$ tokens), KV Cache memory surpasses model weights!
        """)
        
        sim_seq_len = st.select_slider(
            "Context Length (N tokens) for Memory Simulation:",
            options=[2048, 4096, 8192, 16384, 32768, 65536, 131072],
            value=32768,
            key="sim_seq_len"
        )
        
        sim_layers = 32
        sim_heads = 32
        sim_dim = 128
        fp16_bytes = 2
        
        # Standard full attention KV cache size in GB
        kv_full_gb = (2 * sim_layers * sim_heads * sim_dim * sim_seq_len * fp16_bytes) / (1024**3)
        
        # SWA with window=1024 + 4 attention sinks
        swa_window = 1024
        swa_sinks = 4
        effective_tokens = min(sim_seq_len, swa_window + swa_sinks)
        kv_swa_gb = (2 * sim_layers * sim_heads * sim_dim * effective_tokens * fp16_bytes) / (1024**3)
        
        c1, c2 = st.columns(2)
        c1.metric("Full Attention KV Cache", f"{kv_full_gb:.2f} GB VRAM")
        c2.metric(
            f"SWA (Window {swa_window} + {swa_sinks} Sinks)", 
            f"{kv_swa_gb:.2f} GB VRAM", 
            delta=f"-{(1.0 - kv_swa_gb / kv_full_gb) * 100.0:.1f}% VRAM"
        )
        
    with right_impact:
        st.markdown("#### The 'Attention Sink' Discovery")
        st.markdown(r"""
        A critical insight emphasized in modern streaming attention research:
        - Simply evicting old tokens with naive sliding window leads to catastrophic failure.
        - **Why?** Initial tokens (such as `<s>` / BOS) absorb massive attention scores across all heads, acting as **attention sinks** to satisfy the Softmax normalizer ($\sum = 1$).
        - **Solution:** Keeping just the **first 4 sink tokens** alongside the **sliding local window** allows infinite-length streaming with zero perplexity degradation!
        """)
        
        st.markdown("```\nToken Stream: [Sink 0, Sink 1, Sink 2, Sink 3] ... [w-2, w-1, w (Local)]\n```")
        st.caption("Figure: Initial attention sinks combined with sliding window attention for stable streaming.")

    # ----------------------------------------------------------------------------------
    # PyTorch Implementation Code Reference
    # ----------------------------------------------------------------------------------
    st.divider()
    st.subheader("💻 PyTorch Implementation Reference")
    st.markdown("Clean, vectorized PyTorch implementation for generating the three attention masks demonstrated above:")
    
    st.code(r'''import torch

def create_causal_mask(seq_len: int) -> torch.Tensor:
    """Classic Causal Mask (Vaswani et al., 2017): lower-triangular matrix."""
    return torch.tril(torch.ones(seq_len, seq_len, dtype=torch.float32))

def create_sliding_window_mask(seq_len: int, window_size: int) -> torch.Tensor:
    """Sliding Window Mask: banded lower-triangular matrix (max w previous tokens)."""
    causal = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.float32))
    band = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.float32), diagonal=-window_size)
    return causal * band

def create_dilated_sliding_window_mask(seq_len: int, window_size: int, dilation_rate: int = 2) -> torch.Tensor:
    """Dilated Sliding Window: looks back w tokens, skipping every 2nd token."""
    sliding = create_sliding_window_mask(seq_len, window_size)
    row_idx = torch.arange(seq_len).unsqueeze(1)
    col_idx = torch.arange(seq_len).unsqueeze(0)
    distance = row_idx - col_idx
    dilation_condition = (distance % dilation_rate == 0) & (distance >= 0)
    return sliding * dilation_condition.float()

def create_sliding_window_with_sinks_mask(seq_len: int, window_size: int, num_sinks: int = 4) -> torch.Tensor:
    """SWA with Attention Sinks: retains initial sink tokens for streaming stability."""
    mask = create_sliding_window_mask(seq_len, window_size)
    causal = create_causal_mask(seq_len)
    mask[:, :min(num_sinks, seq_len)] = causal[:, :min(num_sinks, seq_len)]
    return mask
''', language="python")


# ======================================================================================
# Standalone Execution Entrypoint
# ======================================================================================

if __name__ == "__main__":
    # Configure page settings when run directly via `streamlit run`
    st.set_page_config(
        page_title="Attention Mechanisms: Causal vs. Sliding Window",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    render_sliding_window()
