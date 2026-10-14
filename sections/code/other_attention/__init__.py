"""
Other Attention Mechanisms Subpackage
Provides implementations and visualizations for Sliding Window Attention,
Dilated Attention, and Classic Causal Attention comparisons.
"""

from .sliding_window import (
    create_causal_mask,
    create_sliding_window_mask,
    create_dilated_sliding_window_mask,
    create_sliding_window_with_sinks_mask,
    calculate_mask_metrics,
    plot_attention_heatmaps,
    render_sliding_window,
)

__all__ = [
    "create_causal_mask",
    "create_sliding_window_mask",
    "create_dilated_sliding_window_mask",
    "create_sliding_window_with_sinks_mask",
    "calculate_mask_metrics",
    "plot_attention_heatmaps",
    "render_sliding_window",
]
