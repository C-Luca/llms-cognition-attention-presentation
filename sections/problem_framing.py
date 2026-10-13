"""
========================================================================================
Modul: Problem Framing
Responsabil: Hermeneanu Ionuț Silviu (Problem Framer)
========================================================================================
Acest fișier este complet izolat. Orice modificări făcute aici vor apărea
în tabul 'Problem Framing' fără a afecta celelalte secțiuni.
========================================================================================
"""

import html
import math

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

DEFAULT_SENTENCE = "The animal didn't cross the street because it was too tired"
MAX_TOKENS = 16

# Chip colours: done / active / waiting / out of reach
CHIP_STYLES = {
    "done": "background:#1e3a5f;color:#e2e8f0;border:1px solid #1e3a5f;",
    "active": "background:#f59e0b;color:#111827;border:1px solid #f59e0b;font-weight:700;",
    "waiting": "background:transparent;color:#94a3b8;border:1px dashed #64748b;",
    "focus": "background:#10b981;color:#0b1220;border:1px solid #10b981;font-weight:700;",
}


def _tokenize(text: str) -> list[str]:
    tokens = text.split()
    return tokens[:MAX_TOKENS] if tokens else DEFAULT_SENTENCE.split()


def _chips(tokens: list[str], states: list[str]) -> str:
    """Render tokens as coloured chips; states index into CHIP_STYLES."""
    parts = []
    for token, state in zip(tokens, states):
        style = CHIP_STYLES[state]
        parts.append(
            f'<span style="{style}padding:4px 10px;margin:3px;border-radius:8px;'
            f'display:inline-block;font-size:1rem;">{html.escape(token)}</span>'
        )
    return '<div style="line-height:2.4;">' + "".join(parts) + "</div>"


def _speaker_notes(text: str) -> None:
    with st.expander("🎤 Speaker notes"):
        st.markdown(text)


# ----------------------------------------------------------------------------------------
# 1. The task
# ----------------------------------------------------------------------------------------
def _render_task() -> None:
    st.subheader("A machine translation paper, not a chatbot paper")
    st.markdown(
        "In 2017 the goal was **sequence transduction**: turn one sequence into another. "
        "The test bed was translation. LLMs like GPT did not exist yet; they were built on this architecture later."
    )

    c1, a1, c2, a2, c3, a3, c4 = st.columns([3, 1, 3, 1, 3, 1, 3])
    boxes = [
        (c1, "English in", "The cat sat on the mat"),
        (c2, "Encoder", "reads the whole source sentence"),
        (c3, "Decoder", "writes the translation one word at a time"),
        (c4, "German out", "Die Katze saß auf der Matte"),
    ]
    for col, title, body in boxes:
        with col:
            with st.container(border=True):
                st.caption(title)
                st.markdown(f"**{body}**")
    for arrow_col in (a1, a2, a3):
        with arrow_col:
            st.markdown("<div style='text-align:center;font-size:2rem;padding-top:1.2rem;'>→</div>", unsafe_allow_html=True)

    st.markdown("#### The benchmark")
    m1, m2, m3 = st.columns(3)
    m1.metric("English→German pairs", "≈4.5M", help="WMT 2014 English–German training set")
    m2.metric("English→French sentences", "36M", help="WMT 2014 English–French training set")
    m3.metric("Quality score", "BLEU", help="0–100: word overlap between the machine translation and human reference translations")

    st.info(
        "The encoder–decoder shape was already standard (Sutskever et al. 2014, Cho et al. 2014). "
        "The Transformer keeps the two boxes and changes **what is inside them**."
    )

    _speaker_notes(
        "- Set the scene: this is a translation paper. The model reads English and writes German or French.\n"
        "- Encoder reads the source, decoder writes the target word by word (auto-regressive: each new word depends on the ones already written).\n"
        "- The paper does not invent this shape. It replaces what is inside the encoder and decoder.\n"
        "- Bridge: *\"In 2017 those boxes were filled with RNNs. So what was wrong with RNNs?\"*"
    )


# ----------------------------------------------------------------------------------------
# 2. The RNN bottleneck
# ----------------------------------------------------------------------------------------
def _render_rnn_bottleneck() -> None:
    st.subheader("RNNs read one word at a time, like a queue")
    st.markdown("A recurrent network updates a memory vector (the *hidden state*) after every word:")
    st.latex(r"h_t = f(h_{t-1},\, x_t)")
    st.markdown("Step *t* needs the result of step *t − 1*, so the words of one sentence can never be processed at the same time.")

    st.markdown("#### Problem 1: no parallelism")
    sentence = st.text_input("Sentence", value=DEFAULT_SENTENCE, key="pf_rnn_sentence")
    tokens = _tokenize(sentence)
    n = len(tokens)

    step = st.slider("Time step", min_value=1, max_value=n, value=1, key="pf_rnn_step") if n > 1 else 1

    left, right = st.columns(2)
    with left:
        st.markdown(f"**RNN: step {step} of {n}**")
        states = ["done" if i < step - 1 else "active" if i == step - 1 else "waiting" for i in range(n)]
        st.markdown(_chips(tokens, states), unsafe_allow_html=True)
        st.caption(f"{n - step} word(s) still waiting. Sequential operations needed: {n}.")
    with right:
        st.markdown("**Self-attention: step 1 of 1**")
        st.markdown(_chips(tokens, ["active"] * n), unsafe_allow_html=True)
        st.caption("All words are processed together. Sequential operations needed: 1.")

    st.markdown("#### Problem 2: far-apart words")
    if n > 1:
        i, j = st.select_slider(
            "Pick two words",
            options=list(range(n)),
            value=(0, n - 1),
            format_func=lambda k: f"{k + 1}: {tokens[k]}",
            key="pf_rnn_pair",
        )
        distance = abs(j - i)
        p1, p2 = st.columns(2)
        p1.metric(f"RNN: steps from '{tokens[i]}' to '{tokens[j]}'", distance)
        p2.metric("Self-attention: steps", 1 if distance else 0)
        st.caption(
            "In an RNN, information from one word reaches another only by passing through every word in between, "
            "and it fades along the way. Shorter paths make long-range dependencies easier to learn (Hochreiter et al., 2001)."
        )

    c1, c2 = st.columns(2)
    with c1:
        with st.container(border=True):
            st.markdown("**Slow**")
            st.markdown("GPUs are built to do thousands of operations at once. During training an RNN leaves them waiting in line.")
    with c2:
        with st.container(border=True):
            st.markdown("**Forgetful**")
            st.markdown("Long-distance links pass through many steps. Tricks like factorized LSTMs and mixture-of-experts sped RNNs up, but *\"the fundamental constraint of sequential computation, however, remains\"* (§1).")

    _speaker_notes(
        "- Move the time-step slider: the RNN lights up one word per step, self-attention lights up everything at once.\n"
        "- Problem 1 (speed): nothing inside a sentence runs in parallel, so training is slow.\n"
        "- Problem 2 (distance): pick 'animal' and 'it'. The RNN needs several steps, self-attention needs one.\n"
        "- The paper frames the problem as **computational**: parallelism and path length, not only accuracy.\n"
        "- Bridge: *\"Before 2017, people had already tried two fixes.\"*"
    )


# ----------------------------------------------------------------------------------------
# 3. Attention before 2017
# ----------------------------------------------------------------------------------------
LINEAGE = [
    ("1997", "LSTM (Hochreiter & Schmidhuber)", "Gated memory for long sequences", "Still one word at a time"),
    ("2014", "Seq2seq (Sutskever et al.; Cho et al.)", "Encoder–decoder for translation", "Whole sentence squeezed into one vector"),
    ("2014", "Bahdanau et al. attention", "Decoder looks back at every source word", "Attention is an add-on; the RNN is still sequential"),
    ("2015", "Luong et al.", "Simpler dot-product attention", "Same RNN backbone"),
    ("2016", "GNMT (Google)", "LSTM + attention in production", "Slow and expensive to train"),
    ("2016", "Decomposable attention (Parikh et al.)", "Attention without an RNN, for entailment", "Not a full translation model"),
    ("2016–17", "ByteNet, ConvS2S", "Convolutions: fully parallel", "Far-apart words need many layers"),
    ("2017", "Transformer", "Attention only: no RNN, no convolution", "→ this paper"),
]


def _render_attention_before() -> None:
    st.subheader("Attention already existed, as a helper to the RNN")

    left, right = st.columns(2)
    with left:
        with st.container(border=True):
            st.caption("2014 · Bahdanau et al. · cross-attention")
            st.markdown("**Sentence → other sentence**")
            st.markdown(
                "While writing each German word, the decoder looks back at the English words and decides which ones matter now. "
                "The RNN around it still reads word by word."
            )
    with right:
        with st.container(border=True):
            st.caption("2017 · Transformer · self-attention")
            st.markdown("**Words → words of the same sentence**")
            st.markdown(
                "Each word looks at every other word in its own sentence, all at once, to understand it. "
                "There is no RNN left."
            )

    st.warning(
        "Common confusion: attention itself was never the sequential part. "
        "It always looked at all positions at once. The **RNN** around it read word by word."
    )
    st.caption("Live example of self-attention resolving 'it' → 'animal': **Code → Classic Attention → Winograd Schema**.")

    st.markdown("#### How we got here")
    df = pd.DataFrame(LINEAGE, columns=["Year", "Work", "Contribution", "Limitation"])
    st.dataframe(df, use_container_width=True, hide_index=True)

    _speaker_notes(
        "- Expected question: *attention existed since 2014, so what is new?*\n"
        "- 2014: attention was a bridge between two sentences, bolted onto an RNN.\n"
        "- 2017: self-attention links words inside the same sentence and replaces the RNN entirely.\n"
        "- The table tells the story: attention starts as a patch on RNNs and ends up replacing them. Point at the last row.\n"
        "- Bridge: *\"The second fix avoided RNNs completely: convolutions.\"*"
    )


# ----------------------------------------------------------------------------------------
# 4. The CNN route
# ----------------------------------------------------------------------------------------
def _cnn_layers_needed(n: int, k: int) -> int:
    """Centered contiguous kernels: each layer extends reach by (k-1)/2 positions per side."""
    reach_per_layer = (k - 1) // 2
    return math.ceil((n - 1) / reach_per_layer) if n > 1 else 0


def _render_cnn_route() -> None:
    st.subheader("CNNs were already parallel, but only saw nearby words")
    st.markdown(
        "A convolution slides a small window over the sentence. Every window can be computed at the same time, "
        "so ByteNet (2016) and ConvS2S (2017) translated **without** an RNN. The catch: one window only sees its neighbours."
    )

    sentence = st.text_input("Sentence", value=DEFAULT_SENTENCE, key="pf_cnn_sentence")
    tokens = _tokenize(sentence)
    n = len(tokens)

    c1, c2, c3 = st.columns(3)
    k = c1.select_slider("Kernel width k", options=[3, 5, 7], value=3, key="pf_cnn_k")
    max_layers = max(1, _cnn_layers_needed(n, k))
    layers = c2.slider("Stacked layers", min_value=1, max_value=max(2, max_layers), value=1, key="pf_cnn_layers")
    focus = c3.selectbox(
        "Focus word",
        options=list(range(n)),
        index=min(7, n - 1),
        format_func=lambda idx: f"{idx + 1}: {tokens[idx]}",
        key="pf_cnn_focus",
    )

    reach = layers * (k - 1) // 2
    z = []
    for layer in range(1, layers + 1):
        r = layer * (k - 1) // 2
        row = [1 if abs(j - focus) <= r else 0 for j in range(n)]
        row[focus] = 2
        z.append(row)

    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=[f"{j + 1}. {t}" for j, t in enumerate(tokens)],
            y=[f"Layer {layer}" for layer in range(1, layers + 1)],
            colorscale=[[0, "#1f2937"], [0.5, "#38bdf8"], [1, "#10b981"]],
            zmin=0,
            zmax=2,
            showscale=False,
            xgap=3,
            ygap=3,
            hovertemplate="%{x}<extra></extra>",
        )
    )
    fig.update_layout(
        height=120 + 45 * layers,
        margin=dict(l=10, r=10, t=30, b=10),
        title=f"What '{tokens[focus]}' can see after each layer (blue = visible)",
    )
    st.plotly_chart(fig, use_container_width=True)

    visible = sum(1 for j in range(n) if abs(j - focus) <= reach)
    needed = _cnn_layers_needed(n, k)
    m1, m2, m3 = st.columns(3)
    m1.metric("Words visible now", f"{visible} / {n}")
    m2.metric(f"CNN layers to link word 1 and word {n}", needed)
    m3.metric("Self-attention layers needed", 1 if n > 1 else 0)

    st.caption(
        "Plain stacked kernels need O(n/k) layers; ByteNet's dilated kernels cut this to O(logₖ n). "
        "Self-attention connects any two words in a single layer."
    )

    _speaker_notes(
        "- Same idea as CNNs for images: a small window slides over the sequence.\n"
        "- Show k = 3 with 1 layer: the focus word sees only its neighbours. Add layers: the visible region grows step by step.\n"
        "- So parallelism alone was **not** new. What CNNs lacked was a short path between distant words.\n"
        "- Bridge: *\"So each option had a weakness. The paper compares them in one table.\"*"
    )


# ----------------------------------------------------------------------------------------
# 5. Comparing the options (Table 1)
# ----------------------------------------------------------------------------------------
TABLE_1 = [
    ("Self-attention", "O(n² · d)", "O(1)", "O(1)"),
    ("Recurrent", "O(n · d²)", "O(n)", "O(n)"),
    ("Convolutional", "O(k · n · d²)", "O(1)", "O(logₖ n)"),
    ("Self-attention (restricted to r neighbours)", "O(r · n · d)", "O(1)", "O(n / r)"),
]


def _render_comparison() -> None:
    st.subheader("Three things a layer should minimise (Table 1)")
    st.markdown(
        "The authors compare layer types on **compute per layer**, **sequential operations** "
        "(how much must wait) and **maximum path length** (steps between two distant words)."
    )
    df = pd.DataFrame(TABLE_1, columns=["Layer type", "Compute per layer", "Sequential ops", "Max path length"])
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.caption("n = sequence length · d = representation size · k = kernel width · r = window size. Source: Vaswani et al. 2017, Table 1.")

    st.markdown("#### The price of self-attention: n²")
    c1, c2 = st.columns(2)
    n = c1.select_slider(
        "Sequence length n (tokens)",
        options=[10, 30, 100, 300, 512, 1000, 3000, 10000, 30000, 100000],
        value=30,
        key="pf_cmp_n",
    )
    d = c2.select_slider("Representation size d", options=[256, 512, 1024, 4096], value=512, key="pf_cmp_d")
    k, r = 3, 64

    costs = {
        "Self-attention": n * n * d,
        "Recurrent": n * d * d,
        "Convolutional (k=3)": k * n * d * d,
        "Restricted self-attention (r=64)": min(r, n) * n * d,
    }
    fig = go.Figure(
        go.Bar(
            x=list(costs.keys()),
            y=list(costs.values()),
            marker_color=["#f59e0b", "#64748b", "#64748b", "#38bdf8"],
            text=[f"{v:.1e}" for v in costs.values()],
            textposition="outside",
            cliponaxis=False,
        )
    )
    fig.update_layout(
        yaxis_type="log",
        yaxis_title="Operations per layer (order of magnitude)",
        height=380,
        margin=dict(l=10, r=10, t=30, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Big-O counts with constants dropped: compare orders of magnitude, not exact runtimes.")

    if n < d:
        st.success(f"n = {n} < d = {d}: self-attention is cheaper than a recurrent layer. This was the 2017 situation (sentences of tens of tokens).")
    elif n == d:
        st.info(f"n = d = {d}: self-attention and recurrence cost about the same.")
    else:
        st.error(f"n = {n:,} > d = {d}: self-attention is now more expensive than recurrence. This is today's long-context problem.")

    m1, m2 = st.columns(2)
    m1.metric("Word pairs to score (n²)", f"{n * n:,}")
    ratio = (n / 30) ** 2
    m2.metric("vs. a 30-word sentence", f"{ratio:,.0f}×" if ratio >= 1 else f"{ratio:.2f}×")

    st.caption(
        "The last row of Table 1, restricted attention, was the authors' own idea for long inputs. "
        "It comes back in 2026 as sliding-window attention: **Code → Other Types Of Attention**."
    )

    _speaker_notes(
        "- Self-attention wins two of the three columns outright: O(1) sequential ops and O(1) path length.\n"
        "- The cost is n² · d. Move n from 30 to 30,000: the number of pairs grows a million times.\n"
        "- Show n < d (green) vs n > d (red): the authors' argument held in 2017 because sentences were short.\n"
        "- Point at the restricted row: the authors already suggested a fix for long inputs; our Code section shows it in 2026.\n"
        "- Bridge: *\"So the authors made a bet: build the whole model from attention.\"*"
    )


# ----------------------------------------------------------------------------------------
# 6. The gap and the bet
# ----------------------------------------------------------------------------------------
def _render_gap_and_bet() -> None:
    st.subheader("The gap, and the bet")
    st.markdown(
        "> *\"To the best of our knowledge, however, the Transformer is the first transduction model relying entirely on "
        "self-attention to compute representations of its input and output without using sequence-aligned RNNs or convolution.\"*  \n"
        "> — Vaswani et al. 2017, §2"
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        with st.container(border=True):
            st.markdown("**Kept**")
            st.markdown("- Encoder–decoder shape\n- Residual connections\n- Layer normalization\n- Embeddings + softmax output")
    with c2:
        with st.container(border=True):
            st.markdown("**Removed**")
            st.markdown("- Recurrence (RNN/LSTM)\n- Convolutions")
    with c3:
        with st.container(border=True):
            st.markdown("**Added to make it work**")
            st.markdown("- Scaled dot-product attention\n- Multi-head attention\n- Positional encodings (word order)")

    st.markdown("#### What the bet had to deliver")
    g1, g2, g3 = st.columns(3)
    g1.markdown("**Parallel training**  \nno sequential steps inside a sentence")
    g2.markdown("**Short paths**  \nany two words linked in one step")
    g3.markdown("**Word order kept**  \nwithout reading in sequence")

    st.info("➡️ How each ingredient works is next, in the **Methodology** tab.")

    _speaker_notes(
        "- Read the quote: this is the paper's exact novelty claim.\n"
        "- Kept / removed / added: the novelty is a subtraction. Attention, residuals and the encoder–decoder already existed.\n"
        "- The three 'added' pieces are what make attention-only possible; don't explain them here.\n"
        "- Hand over: *\"Teodor and Luca will now show how these pieces work inside.\"*"
    )


# ----------------------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------------------
def render_problem_framing():
    st.header("Problem Framing")
    st.caption("Why the Transformer was needed: the problem, what existed before, and the gap the paper fills. ≈10 min.")

    tabs = st.tabs([
        "① The Task",
        "② The RNN Bottleneck",
        "③ Attention Before 2017",
        "④ The CNN Route",
        "⑤ Comparing the Options",
        "⑥ The Gap & The Bet",
    ])
    renderers = [
        _render_task,
        _render_rnn_bottleneck,
        _render_attention_before,
        _render_cnn_route,
        _render_comparison,
        _render_gap_and_bet,
    ]
    for tab, render in zip(tabs, renderers):
        with tab:
            render()
