"""
========================================================================================
Module: Results - paper data
Author: Gusev Roman (Results Interpreter & Challenger)
========================================================================================
Every number below is transcribed from Vaswani et al. (2017), "Attention Is All You Need":
Table 2 (Sec. 6.1), Table 3 (Sec. 6.2) and the training details of Sec. 5.
Nothing here is re-run or estimated: the tabs only visualise and interpret these values.
========================================================================================
"""

import pandas as pd

# Colours shared by every results tab (same palette as the Problem Framing section)
AMBER = "#f59e0b"   # the Transformer / the thing to look at
SLATE = "#64748b"   # previous work
BLUE = "#38bdf8"    # secondary highlight
GREEN = "#10b981"   # positive / good
DARK = "#475569"    # ensembles


# ----------------------------------------------------------------------------------------
# Table 2: BLEU on newstest2014 + training cost (FLOPs)
# kind: "single" | "ensemble" | "transformer". None = not reported in the paper.
# The paper prints one cost cell for the Transformer rows; it applies to both language pairs.
# ----------------------------------------------------------------------------------------
TABLE_2 = [
    # model,                         kind,          BLEU DE, BLEU FR, FLOPs DE, FLOPs FR
    ("ByteNet",                      "single",      23.75, None,  None,   None),
    ("Deep-Att + PosUnk",            "single",      None,  39.2,  None,   1.0e20),
    ("GNMT + RL",                    "single",      24.6,  39.92, 2.3e19, 1.4e20),
    ("ConvS2S",                      "single",      25.16, 40.46, 9.6e18, 1.5e20),
    ("MoE",                          "single",      26.03, 40.56, 2.0e19, 1.2e20),
    ("Deep-Att + PosUnk Ensemble",   "ensemble",    None,  40.4,  None,   8.0e20),
    ("GNMT + RL Ensemble",           "ensemble",    26.30, 41.16, 1.8e20, 1.1e21),
    ("ConvS2S Ensemble",             "ensemble",    26.36, 41.29, 7.7e19, 1.2e21),
    ("Transformer (base model)",     "transformer", 27.3,  38.1,  3.3e18, 3.3e18),
    ("Transformer (big)",            "transformer", 28.4,  41.8,  2.3e19, 2.3e19),
]
TABLE_2_COLUMNS = ["Model", "Kind", "BLEU EN-DE", "BLEU EN-FR", "FLOPs EN-DE", "FLOPs EN-FR"]


def table_2_df() -> pd.DataFrame:
    return pd.DataFrame(TABLE_2, columns=TABLE_2_COLUMNS)


# ----------------------------------------------------------------------------------------
# Table 3: model variations, EN-DE development set (newstest2013)
# Unlisted values are identical to the base model (N=6, d_model=512, d_ff=2048, h=8,
# d_k=d_v=64, P_drop=0.1, eps_ls=0.1, 100K steps). PPL is per word-piece.
# ----------------------------------------------------------------------------------------
BASE_CONFIG = dict(N=6, d_model=512, d_ff=2048, h=8, d_k=64, d_v=64, p_drop=0.1, eps_ls=0.1, steps="100K")

_TABLE_3_ROWS = [
    # group, label,                              overrides,                            PPL,  BLEU, params(M)
    ("base", "base",                             {},                                    4.92, 25.8, 65),
    ("A",    "h=1 (d_k=d_v=512)",                dict(h=1, d_k=512, d_v=512),           5.29, 24.9, 65),
    ("A",    "h=4 (d_k=d_v=128)",                dict(h=4, d_k=128, d_v=128),           5.00, 25.5, 65),
    ("A",    "h=16 (d_k=d_v=32)",                dict(h=16, d_k=32, d_v=32),            4.91, 25.8, 65),
    ("A",    "h=32 (d_k=d_v=16)",                dict(h=32, d_k=16, d_v=16),            5.01, 25.4, 65),
    ("B",    "d_k=16",                           dict(d_k=16),                          5.16, 25.1, 58),
    ("B",    "d_k=32",                           dict(d_k=32),                          5.01, 25.4, 60),
    ("C",    "N=2 layers",                       dict(N=2),                             6.11, 23.7, 36),
    ("C",    "N=4 layers",                       dict(N=4),                             5.19, 25.3, 50),
    ("C",    "N=8 layers",                       dict(N=8),                             4.88, 25.5, 80),
    ("C",    "d_model=256",                      dict(d_model=256, d_k=32, d_v=32),     5.75, 24.5, 28),
    ("C",    "d_model=1024",                     dict(d_model=1024, d_k=128, d_v=128),  4.66, 26.0, 168),
    ("C",    "d_ff=1024",                        dict(d_ff=1024),                       5.12, 25.4, 53),
    ("C",    "d_ff=4096",                        dict(d_ff=4096),                       4.75, 26.2, 90),
    ("D",    "P_drop=0.0",                       dict(p_drop=0.0),                      5.77, 24.6, 65),
    ("D",    "P_drop=0.2",                       dict(p_drop=0.2),                      4.95, 25.5, 65),
    ("D",    "eps_ls=0.0",                       dict(eps_ls=0.0),                      4.67, 25.3, 65),
    ("D",    "eps_ls=0.2",                       dict(eps_ls=0.2),                      5.47, 25.7, 65),
    ("E",    "learned positional embedding",     {},                                    4.92, 25.7, 65),
    ("big",  "big",                              dict(d_model=1024, d_ff=4096, h=16, p_drop=0.3, steps="300K"), 4.33, 26.4, 213),
]

GROUP_TITLES = {
    "base": "Base model",
    "A": "(A) Number of attention heads",
    "B": "(B) Attention key size d_k",
    "C": "(C) Model size: layers, d_model, d_ff",
    "D": "(D) Regularization: dropout and label smoothing",
    "E": "(E) Sinusoidal vs learned positions",
    "big": "Big model",
}


def table_3_df() -> pd.DataFrame:
    rows = []
    for group, label, overrides, ppl, bleu, params in _TABLE_3_ROWS:
        cfg = {**BASE_CONFIG, **overrides}
        rows.append({"Group": group, "Variation": label, **cfg, "PPL (dev)": ppl, "BLEU (dev)": bleu, "Params (M)": params})
    return pd.DataFrame(rows).rename(columns={
        "N": "N", "d_model": "d_model", "d_ff": "d_ff", "h": "h", "d_k": "d_k", "d_v": "d_v",
        "p_drop": "P_drop", "eps_ls": "eps_ls", "steps": "Train steps",
    })


# ----------------------------------------------------------------------------------------
# Sustained single-precision throughput the authors assumed per GPU (footnote 5, TFLOPS)
# ----------------------------------------------------------------------------------------
GPU_TFLOPS = {"K80": 2.8, "K40": 3.7, "M40": 6.0, "P100": 9.5}
