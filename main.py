"""
========================================================================================
LLMs & Cognition 2026 - UniBuc FMI
Prezentare: "Attention Is All You Need" (Vaswani et al., 2017)
Articol Corelat: "Sliding-Window Beats Linear Attention" (Jolicoeur-Martineau et al., 2026)
Master Application (Arhitectură Modulară pe Fișiere Separate)
========================================================================================
"""

import os
import sys
import streamlit as st
import plotly.io as pio
import matplotlib.pyplot as plt

# Asigură că directorul aplicației este în sys.path pentru importuri robuste
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# Importuri modulare din fișiere separate
from sections.problem_framing import render_problem_framing
from sections.methodology import render_methodology

from sections.results.experimental_setup import render_experimental_setup
from sections.results.complexity_analysis import render_complexity_analysis
from sections.results.machine_translation import render_machine_translation
from sections.results.model_variations import render_model_variations
from sections.results.discussion_conclusion import render_discussion_conclusion
from sections.results.swa_paper import render_swa_paper

from sections.code.classic_attention.code_paradigms import render_code_paradigms
from sections.code.classic_attention.scaled_dot import render_scaled_dot
from sections.code.classic_attention.winograd import render_winograd
from sections.code.classic_attention.shape_inspector import render_shape_inspector
from sections.code.classic_attention.bertviz_view import render_bertviz_view

from sections.code.other_attention.sliding_window import render_sliding_window

# Configurare pagină Streamlit
st.set_page_config(
    page_title="Attention Is All You Need",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Setare temă implicită Plotly și Matplotlib
pio.templates.default = "plotly_dark"
try:
    plt.style.use("dark_background")
except Exception:
    pass

# Stiluri custom CSS
st.markdown("""
<style>
    :root {
        --bg-main: #0b0f19;
        --bg-card: #131b2e;
        --bg-card-subtle: #182238;
        --border-subtle: #263554;
        --border-active: #3b82f6;
        --text-primary: #f8fafc;
        --text-secondary: #94a3b8;
        --text-muted: #64748b;
        --tab-bg: #131b2e;
        --tab-border: #263554;
        --tab-text: #94a3b8;
        --tab-active-bg: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        --tab-active-text: #ffffff;
        --tab-active-shadow: 0 3px 12px rgba(37, 99, 235, 0.4);
        --subtab-bg: #0b0f19;
        --subtab-border: #1e293b;
        --subtab-text: #94a3b8;
        --subtab-active-bg: linear-gradient(135deg, #0284c7 0%, #0369a1 100%);
        --metric-bg: #131b2e;
        --code-bg: #080c14;
    }

    /* Ascunde meniul lateral complet */
    [data-testid="stSidebar"] {
        display: none !important;
    }

    /* Padding adaptat astfel încât bara Streamlit de sus (deploy / meniu) să nu acopere titlul */
    [data-testid="stMainBlockContainer"] {
        padding-top: 4.8rem !important;
        padding-bottom: 2rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        max-width: 98% !important;
    }

    /* Background și text global */
    .stApp, [data-testid="stAppViewContainer"], .main {
        background-color: var(--bg-main) !important;
        color: var(--text-primary) !important;
    }
    [data-testid="stHeader"] {
        background-color: transparent !important;
    }

    /* Carduri și containere cu border */
    [data-testid="stVerticalBlockBorderWrapper"] {
        background-color: var(--bg-card) !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: 12px !important;
    }

    /* Typography */
    .stMarkdown, p, span, li, label {
        color: var(--text-primary) !important;
    }
    h1, h2, h3, h4, h5, h6 {
        color: var(--text-primary) !important;
    }

    /* Titlu principal perfect vizibil, clar și fără clipping */
    .main-title {
        font-size: 2.35rem !important;
        font-weight: 800 !important;
        color: #ffffff !important;
        letter-spacing: -0.025em !important;
        line-height: 1.25 !important;
        margin-bottom: 0.25rem !important;
        text-shadow: 0 2px 10px rgba(0, 0, 0, 0.4);
    }
    .sub-title {
        color: var(--text-secondary) !important;
        font-size: 0.96rem !important;
        margin-bottom: 1.1rem !important;
        letter-spacing: -0.01em !important;
    }

    /* =========================================================================
       SISTEM DE NAVIGARE COMPACT ȘI ELEGANT (Tabs ca Pills/Segmented Controls)
       ========================================================================= */

    /* Nivel 1: Meniu principal */
    .stTabs {
        margin-top: 0rem !important;
        margin-bottom: 0.4rem !important;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px !important;
        background: var(--tab-bg) !important;
        padding: 4px 6px !important;
        border-radius: 12px !important;
        border: 1px solid var(--tab-border) !important;
        display: inline-flex !important;
        width: auto !important;
        max-width: 100% !important;
        overflow-x: auto !important;
        margin-bottom: 0.5rem !important;
    }
    .stTabs [data-baseweb="tab"] {
        height: 32px !important;
        padding: 3px 14px !important;
        border-radius: 8px !important;
        font-size: 0.88rem !important;
        font-weight: 500 !important;
        color: var(--tab-text) !important;
        background: transparent !important;
        border: 1px solid transparent !important;
        transition: all 0.15s ease-in-out !important;
        white-space: nowrap !important;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: var(--text-primary) !important;
        background: rgba(148, 163, 184, 0.12) !important;
    }
    .stTabs [aria-selected="true"] {
        background: var(--tab-active-bg) !important;
        color: var(--tab-active-text) !important;
        font-weight: 600 !important;
        border-color: transparent !important;
        box-shadow: var(--tab-active-shadow) !important;
    }
    .stTabs [data-baseweb="tab-highlight"],
    .stTabs [data-baseweb="tab-border"] {
        display: none !important;
    }

    /* Nivel 2 și 3: Submeniuri imbricate (Pills mai fine și discrete) */
    .stTabs .stTabs [data-baseweb="tab-list"] {
        gap: 4px !important;
        background: var(--subtab-bg) !important;
        padding: 3px 4px !important;
        border-radius: 10px !important;
        border: 1px solid var(--subtab-border) !important;
        margin-bottom: 0.5rem !important;
    }
    .stTabs .stTabs [data-baseweb="tab"] {
        height: 27px !important;
        padding: 2px 11px !important;
        border-radius: 6px !important;
        font-size: 0.82rem !important;
        color: var(--subtab-text) !important;
    }
    .stTabs .stTabs [aria-selected="true"] {
        background: var(--subtab-active-bg) !important;
        color: #ffffff !important;
        box-shadow: 0 2px 6px rgba(2, 132, 199, 0.3) !important;
    }

    /* Componente UI: Metrics, Expanders, Input-uri */
    [data-testid="stMetric"] {
        background-color: var(--metric-bg) !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: 10px !important;
        padding: 8px 12px !important;
    }
    [data-testid="stMetricValue"] {
        color: var(--text-primary) !important;
    }
    [data-testid="stMetricLabel"] {
        color: var(--text-secondary) !important;
    }

    [data-testid="stExpander"] {
        background-color: var(--bg-card) !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: 10px !important;
    }

    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div {
        background-color: var(--bg-card) !important;
        border-color: var(--border-subtle) !important;
        color: var(--text-primary) !important;
    }

    /* Butoane radio inline curate */
    div[data-testid="stRadio"] > div {
        gap: 12px;
    }
</style>
""", unsafe_allow_html=True)

# Header aplicație (Full width, clar și elegant)
st.markdown('<div class="main-title">Attention Is All You Need</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Hermeneanu Ionuț Silviu • Leahu Teodor Filip • Szocs Luca • Cuclea Luca-Nicolae • Gheorghe Bogdan-Alexandru • Gusev Roman</div>', unsafe_allow_html=True)

# ==============================================================================
# NIVEL 1: MENIURI PRINCIPALE (Tabs stilizate compact)
# ==============================================================================
main_pf, main_me, main_code, main_res = st.tabs([
    "Problem Framing",
    "Methodology",
    "Code",
    "Results"
])

# 1. Problem Framing (Ionuț)
with main_pf:
    try:
        render_problem_framing()
    except Exception as e:
        st.error(f"Error in Problem Framing module: {e}")

# 2. Methodology (Teodor & Luca Szocs)
with main_me:
    try:
        render_methodology()
    except Exception as e:
        st.error(f"Error in Methodology module: {e}")

# 3. Code (Cuclea Luca & Bogdan)
with main_code:
    # NIVEL 2: Meniu unificat direct pe module (fără imbricare excesivă)
    (
        tab_paradigms,
        tab_scaled,
        tab_winograd,
        tab_shapes,
        tab_bertviz,
        tab_swa,
    ) = st.tabs([
        "Code Comparison: 3 Paradigms",
        "Scaled Dot-Product & Saturation",
        "Winograd Schema & Cognition",
        "PyTorch Shape Inspector",
        "BertViz Multiscale Explorer",
        "Sliding-Window Attention & Sinks",
    ])

    with tab_paradigms:
        render_code_paradigms()

    with tab_scaled:
        render_scaled_dot()

    with tab_winograd:
        render_winograd()

    with tab_shapes:
        render_shape_inspector()

    with tab_bertviz:
        render_bertviz_view()

    with tab_swa:
        render_sliding_window()

# 4. Results (Roman)
with main_res:
    # NIVEL 2: SUBMENIURI SUB RESULTS
    (
        res_setup,
        res_complexity,
        res_mt,
        res_variations,
        res_discussion,
        res_swa,
    ) = st.tabs([
        "Experimental Setup",
        "Complexity Analysis",
        "Machine Translation",
        "Model Variations",
        "Discussion & Conclusion",
        "Sliding-Window vs Linear Attention (2026)",
    ])

    for tab, render, name in [
        (res_setup, render_experimental_setup, "Experimental Setup"),
        (res_complexity, render_complexity_analysis, "Complexity Analysis"),
        (res_mt, render_machine_translation, "Machine Translation"),
        (res_variations, render_model_variations, "Model Variations"),
        (res_discussion, render_discussion_conclusion, "Discussion & Conclusion"),
        (res_swa, render_swa_paper, "Sliding-Window vs Linear Attention"),
    ]:
        with tab:
            try:
                render()
            except Exception as e:
                st.error(f"Error in Results module ({name}): {e}")
