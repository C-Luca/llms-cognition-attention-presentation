"""
Other Attention Mechanisms Subpackage
Presentation module for "Sliding-window beats linear attention" (arXiv:2608.28444).
"""

from .sliding_window import (
    create_causal_mask,
    create_naive_sliding_window_mask,
    create_swa_paper_mask,
    create_sliding_window_mask,
    create_dilated_sliding_window_mask,
    create_sliding_window_with_sinks_mask,
    calculate_paper_mask_metrics,
    calculate_mask_metrics,
    plot_paper_attention_heatmaps,
    plot_attention_heatmaps,
    plot_flashattention_comparison,
    render_sliding_window,
)

__all__ = [
    "create_causal_mask",
    "create_naive_sliding_window_mask",
    "create_swa_paper_mask",
    "create_sliding_window_mask",
    "create_dilated_sliding_window_mask",
    "create_sliding_window_with_sinks_mask",
    "calculate_paper_mask_metrics",
    "calculate_mask_metrics",
    "plot_paper_attention_heatmaps",
    "plot_attention_heatmaps",
    "plot_flashattention_comparison",
    "render_sliding_window",
]
