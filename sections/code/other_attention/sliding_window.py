"""
========================================================================================
Modul: Sliding Window Attention & Attention Sinks (Paper-ul din 2026)
Responsabil: Cuclea Luca-Nicolae & Gheorghe Bogdan-Alexandru (Coder)
========================================================================================
"""

import numpy as np
import streamlit as st
import plotly.express as px

def render_sliding_window():
    st.header("Sliding Window Attention & Attention Sinks: Articolul din 2026")
    
    col_swa_left, col_swa_right = st.columns([1, 1])
    
    with col_swa_left:
        st.subheader("Problema Memoriei: KV Cache Explosion")
        st.markdown(r"""
        La generarea textului pas cu pas (inference), stocăm în memoria GPU toți vectorii $K$ și $V$ precedenți (**KV Cache**):
        - **Full Attention:** Memoria crește liniar cu numărul de tokeni $\mathcal{O}(N)$, iar operațiile $\mathcal{O}(N^2)$.
        - La un context de 32k sau 128k tokeni, memoria KV Cache depășește memoria parametrilor modelului!
        """)
        
        # Interactive KV Cache Calculator
        st.markdown("#### 🧮 Calculator Interactiv de Memorie KV Cache")
        c_seq_len = st.select_slider("Lungime Context (N tokeni)", options=[1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072], value=32768)
        c_layers = 32
        c_heads = 32
        c_head_dim = 128
        precision_bytes = 2 # FP16
        
        # Formula KV Cache: 2 (K si V) * layers * heads * head_dim * seq_len * bytes
        kv_cache_full_gb = (2 * c_layers * c_heads * c_head_dim * c_seq_len * precision_bytes) / (1024**3)
        
        # Sliding Window cu window=1024 + 4 sinks
        window_size = 1024
        sinks = 4
        effective_swa_tokens = min(c_seq_len, window_size + sinks)
        kv_cache_swa_gb = (2 * c_layers * c_heads * c_head_dim * effective_swa_tokens * precision_bytes) / (1024**3)
        
        m_c1, m_c2 = st.columns(2)
        m_c1.metric("Full Attention KV Cache", f"{kv_cache_full_gb:.2f} GB VRAM")
        m_c2.metric(f"SWA (Window {window_size} + Sinks)", f"{kv_cache_swa_gb:.2f} GB VRAM", delta=f"-{(1 - kv_cache_swa_gb/kv_cache_full_gb)*100:.1f}%")

    with col_swa_right:
        st.subheader("Ce este un 'Attention Sink'?")
        st.markdown("""
        Dacă tăiem doar fereastra glisantă simplă (naive SWA), modelul eșuează lamentabil.
        **Descoperirea cheie:** Primele 4 tokeni dintr-o secvență (cum ar fi token-ul `<s>` / BOS) absorb ponderi imense de atenție, chiar dacă nu au sens semantic! 
        Ele funcționează ca un canal de scurgere (*sink*) pentru suma softmax-ului ($=1$).
        
        Dacă păstrăm **primele $S$ tokeni** + **ultimii $W$ tokeni locali**, modelul rămâne stabil la infinit fără niciun re-antrenament!
        """)
        
        # Mask Visualizer
        mask_type = st.radio("Tip Mască Atenție:", ["1. Full Causal Attention", "2. Naive Sliding Window (fără sinks)", "3. SWA cu Attention Sinks (Articolul 2026)"], horizontal=True)
        
        n_vis = 16
        w_vis = 4
        s_vis = 2
        mask_mat = np.zeros((n_vis, n_vis))
        
        for i in range(n_vis):
            for j in range(n_vis):
                if j <= i: # causal condition
                    if mask_type.startswith("1"):
                        mask_mat[i, j] = 1.0
                    elif mask_type.startswith("2"):
                        if i - j < w_vis:
                            mask_mat[i, j] = 1.0
                    else:
                        if j < s_vis or (i - j < w_vis):
                            mask_mat[i, j] = 1.0
                            
        fig_mask = px.imshow(
            mask_mat,
            color_continuous_scale=[[0, "#0f172a"], [1, "#38bdf8"]],
            title=f"Masca de Atenție (16x16) pentru: {mask_type}",
            labels=dict(x="Key Index", y="Query Index")
        )
        fig_mask.update_layout(height=320, coloraxis_showscale=False)
        st.plotly_chart(fig_mask, use_container_width=True)

    st.divider()
    st.markdown("#### 💻 Cum arată implementarea în cod PyTorch pentru SWA + Sinks:")
    st.code("""def sliding_window_with_sinks_mask(seq_len, window_size=1024, num_sinks=4):
    # 1. Mască triunghiulară cauzală
    mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
    
    # 2. Mască de bandă glisantă (tokenii prea îndepărtați devin False)
    band_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=-(window_size - 1))
    swa_mask = mask & band_mask
    
    # 3. Păstrează primele num_sinks coloane (Attention Sinks) active pentru tot contextul!
    swa_mask[:, :num_sinks] = mask[:, :num_sinks]
    
    return swa_mask # Shape: [seq_len, seq_len]
""", language="python")
