"""
========================================================================================
Module: Winograd Schema & Cognitive Coreference Resolution
Author: Cuclea Luca-Nicolae & Gheorghe Bogdan-Alexandru (Coder)
========================================================================================
"""

import streamlit as st
import plotly.graph_objects as go

def render_winograd():
    st.header("Cognitive Study: Coreference Resolution in Winograd Schemas")
    st.markdown("""
    The Winograd Schema Challenge is a classic benchmark in AI and Computational Linguistics that tests whether a model understands common-sense reasoning and world knowledge, rather than relying solely on superficial grammar.
    """)
    
    w_scenario = st.radio(
        "Select Winograd Scenario:",
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
        explanation = "The adjective **'tired'** is a biological property of an animal, not of a street. The semantic-anaphoric attention head binds the pronoun **'it'** directly to **'animal'** with an 86.4% attention weight."
    elif "WIDE" in w_scenario:
        tokens = ["The", "animal", "didn't", "cross", "the", "street", "because", "it", "was", "too", "wide"]
        query_idx = 7
        weights = [0.01, 0.04, 0.01, 0.02, 0.01, 0.88, 0.01, 0.01, 0.01, 0.01, 0.00]
        weights[10] = 1.0 - sum(weights[:10])
        target = "street"
        explanation = "The adjective **'wide'** is a spatial property of the street. Therefore, attention shifts heavily from the animal to **'street'** (88.1%), resolving the pronoun **'it'**."
    elif "BIG" in w_scenario:
        tokens = ["The", "trophy", "didn't", "fit", "into", "the", "suitcase", "because", "it", "was", "too", "big"]
        query_idx = 8
        weights = [0.01, 0.84, 0.01, 0.02, 0.01, 0.01, 0.06, 0.01, 0.01, 0.01, 0.01, 0.00]
        weights[11] = 1.0 - sum(weights[:11])
        target = "trophy"
        explanation = "If the item does not fit because it is too **'big'**, then the item to be inserted is the large one → **'trophy'**."
    else:
        tokens = ["The", "trophy", "didn't", "fit", "into", "the", "suitcase", "because", "it", "was", "too", "small"]
        query_idx = 8
        weights = [0.01, 0.05, 0.01, 0.02, 0.01, 0.01, 0.85, 0.01, 0.01, 0.01, 0.01, 0.00]
        weights[11] = 1.0 - sum(weights[:11])
        target = "suitcase"
        explanation = "If it does not fit because it is too **'small'**, then the container is the small one → **'suitcase'**."

    with col_w_info:
        st.subheader("Linguistic & Cognitive Analysis")
        st.markdown(f"**Queried Pronoun (Query):** `{tokens[query_idx]}`")
        st.markdown(f"**Resolved Referent (Target):** :green[**{target.upper()}**]")
        st.info(explanation)

    with col_w_chart:
        st.subheader(f"Attention Distribution for Token '{tokens[query_idx]}'")
        fig_bar = go.Figure(go.Bar(
            x=tokens,
            y=weights,
            marker_color=['#6366f1' if t == target else ('#f59e0b' if t == tokens[query_idx] else '#475569') for t in tokens],
            text=[f"{w*100:.1f}%" if w > 0.03 else "" for w in weights],
            textposition='auto',
        ))
        fig_bar.update_layout(
            xaxis_title="Tokens in Sentence",
            yaxis_title="Softmax Attention Weight",
            yaxis=dict(range=[0, 1.0]),
            height=380
        )
        st.plotly_chart(fig_bar, use_container_width=True)
