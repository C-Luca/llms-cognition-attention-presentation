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

# Stiluri custom CSS
st.markdown("""
<style>
    /* Ascunde meniul lateral complet */
    [data-testid="stSidebar"] {
        display: none;
    }
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #6366f1, #38bdf8, #10b981);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        color: #94a3b8;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Header aplicație
st.markdown('<div class="main-title">Attention Is All You Need</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Hermeneanu Ionuț Silviu • Leahu Teodor Filip • Szocs Luca • Cuclea Luca-Nicolae • Gheorghe Bogdan-Alexandru • Gusev Roman</div>', unsafe_allow_html=True)

# ==============================================================================
# NIVEL 1: MENIURI PRINCIPALE (Tabs)
# ==============================================================================
main_pf, main_me, main_code, main_res = st.tabs([
    "📌 Problem Framing",
    "🔬 Methodology",
    "💻 Code",
    "📊 Results"
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

# 4. Results (Roman)
with main_res:
    # NIVEL 2: SUBMENIURI SUB RESULTS
    (res_setup, res_complexity, res_mt, res_variations, res_discussion, res_swa) = st.tabs([
        "🧪 Experimental Setup",
        "⏱️ Complexity Analysis",
        "🌍 Machine Translation",
        "🔬 Model Variations",
        "🏁 Discussion & Conclusion",
        "🌊 Sliding-Window vs Linear Attention (2026)"
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

# 3. Code (Cuclea Luca & Bogdan)
with main_code:
    # NIVEL 2: SUBMENIURI SUB CODE
    sub_classic, sub_other = st.tabs([
        "🏛️ Classic Attention",
        "🌊 Other Types Of Attention"
    ])
    
    # --------------------------------------------------------------------------
    # NIVEL 3: SUB-SUBMENIURI SUB CLASSIC ATTENTION
    # --------------------------------------------------------------------------
    with sub_classic:
        tab_paradigms, tab_scaled, tab_winograd, tab_shapes, tab_bertviz = st.tabs([
            "💻 Code Comparison: 3 Paradigms",
            "⚡ Scaled Dot-Product & Saturation",
            "🧠 Winograd Schema & Cognition",
            "🛠️ PyTorch Shape Inspector",
            "🔍 BertViz Multiscale Explorer"
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

    # --------------------------------------------------------------------------
    # NIVEL 3: OTHER TYPES OF ATTENTION (SLIDING WINDOW & SINKS)
    # --------------------------------------------------------------------------
    with sub_other:
        render_sliding_window()
