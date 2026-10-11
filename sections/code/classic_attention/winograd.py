"""
========================================================================================
Modul: Winograd Schema & Rezolvarea Coreferinței Cognitive
Responsabil: Cuclea Luca-Nicolae & Gheorghe Bogdan-Alexandru (Coder)
========================================================================================
"""

import streamlit as st
import plotly.graph_objects as go

def render_winograd():
    st.header("Studiu Cognitiv: Rezolvarea Coreferinței în Schema Winograd")
    st.markdown("""
    Schema Winograd este un benchmark celebru din AI și Lingvistică Computațională care testează dacă un model înțelege contextul comun (common sense reasoning), nu doar gramatica la suprafață.
    """)
    
    w_scenario = st.radio(
        "Alege Scenariul Winograd:",
        [
            "1. The animal didn't cross the street because IT was too TIRED.",
            "2. The animal didn't cross the street because IT was too WIDE.",
            "3. The trophy did not fit into the brown suitcase because IT was too BIG.",
            "4. The trophy did not fit into the brown suitcase because IT was too SMALL."
        ]
    )
    
    col_w_info, col_w_chart = st.columns([1, 1])
    
    if "TIRED" in w_scenario:
        tokens = ["The", "animal", "didn't", "cross", "the", "street", "because", "it", "was", "too", "tired"]
        query_idx = 7
        weights = [0.01, 0.86, 0.01, 0.02, 0.01, 0.05, 0.01, 0.01, 0.01, 0.01, 0.00]
        weights[10] = 1.0 - sum(weights[:10])
        target = "animal"
        explanation = "Adjectivul **'tired'** (obosit) este o proprietate biologică a unui animal, nu a unei străzi. Capul de atenție semantico-anaforică leagă pronumele **'it'** direct de **'animal'** cu o pondere de 86.4%."
    elif "WIDE" in w_scenario:
        tokens = ["The", "animal", "didn't", "cross", "the", "street", "because", "it", "was", "too", "wide"]
        query_idx = 7
        weights = [0.01, 0.04, 0.01, 0.02, 0.01, 0.88, 0.01, 0.01, 0.01, 0.01, 0.00]
        weights[10] = 1.0 - sum(weights[:10])
        target = "street"
        explanation = "Adjectivul **'wide'** (lat) este o proprietate spațială a străzii. Prin urmare, atenția comută masiv de la animal la **'street'** (88.1%), rezolvând ambiguitatea pronumelui **'it'**."
    elif "BIG" in w_scenario:
        tokens = ["The", "trophy", "didn't", "fit", "into", "the", "suitcase", "because", "it", "was", "too", "big"]
        query_idx = 8
        weights = [0.01, 0.84, 0.01, 0.02, 0.01, 0.01, 0.06, 0.01, 0.01, 0.01, 0.01, 0.00]
        weights[11] = 1.0 - sum(weights[:11])
        target = "trophy"
        explanation = "Dacă obiectul nu încape pentru că este prea **'big'** (mare), atunci obiectul care trebuia introdus este cel mare → **'trophy'**."
    else:
        tokens = ["The", "trophy", "didn't", "fit", "into", "the", "suitcase", "because", "it", "was", "too", "small"]
        query_idx = 8
        weights = [0.01, 0.05, 0.01, 0.02, 0.01, 0.01, 0.85, 0.01, 0.01, 0.01, 0.01, 0.00]
        weights[11] = 1.0 - sum(weights[:11])
        target = "suitcase"
        explanation = "Dacă nu încape pentru că este prea **'small'** (mică), atunci recipientul este cel mic → **'suitcase'**."

    with col_w_info:
        st.subheader("Analiză Lingvistică & Cognitivă")
        st.markdown(f"**Pronume interogat (Query):** `{tokens[query_idx]}`")
        st.markdown(f"**Referent Rezolvat (Target):** :green[**{target.upper()}**]")
        st.info(explanation)

    with col_w_chart:
        st.subheader(f"Distribuția Atenției pentru Token-ul '{tokens[query_idx]}'")
        fig_bar = go.Figure(go.Bar(
            x=tokens,
            y=weights,
            marker_color=['#6366f1' if t == target else ('#f59e0b' if t == tokens[query_idx] else '#475569') for t in tokens],
            text=[f"{w*100:.1f}%" if w > 0.03 else "" for w in weights],
            textposition='auto',
        ))
        fig_bar.update_layout(
            xaxis_title="Tokeni în Propoziție",
            yaxis_title="Pondere Atenție Softmax",
            yaxis=dict(range=[0, 1.0]),
            height=380
        )
        st.plotly_chart(fig_bar, use_container_width=True)
