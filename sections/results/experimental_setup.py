"""
========================================================================================
Module: Results - Experimental Setup
Author: Gusev Roman (Results Interpreter & Challenger)
========================================================================================
Paper Sec. 5 (Training) and Sec. 6.1 (decoding): what the results are measured on, with which
metrics, and how the models were trained. Read this tab first: every later number
(BLEU, perplexity, FLOPs) is defined here.
========================================================================================
"""

import math
import re
from collections import Counter

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from sections.results import wmt_samples
from sections.results.common import challenger_box, md_table, speaker_notes
from sections.results.paper_data import AMBER, BLUE, GREEN, SLATE


# ----------------------------------------------------------------------------------------
# BLEU (sentence-level toy implementation, for explanation only)
# ----------------------------------------------------------------------------------------
def _tokens(text: str) -> list[str]:
    return re.findall(r"\w+|[^\w\s]", text.lower())


def _ngram_counts(tokens: list[str], n: int) -> Counter:
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def toy_bleu(reference: str, hypothesis: str, max_n: int = 4) -> dict:
    """BLEU = brevity penalty x geometric mean of modified n-gram precisions (n = 1..4).

    Sentence-level with a simple smoothing (an n-gram order with zero matches gets 1 / (2 * total)),
    otherwise a single missing 4-gram would zero the whole score. Real evaluations pool counts
    over the entire test set (corpus-level BLEU), which does not need this trick.
    """
    ref, hyp = _tokens(reference), _tokens(hypothesis)
    precisions, rows = [], []
    for n in range(1, max_n + 1):
        hyp_counts, ref_counts = _ngram_counts(hyp, n), _ngram_counts(ref, n)
        total = sum(hyp_counts.values())
        # "modified" precision: a hypothesis n-gram is credited at most as often as it occurs in the reference
        matched = sum(min(count, ref_counts[gram]) for gram, count in hyp_counts.items())
        if total == 0:
            p = 0.0
        elif matched == 0:
            p = 1 / (2 * total)
        else:
            p = matched / total
        precisions.append(p)
        rows.append({"n-gram": f"{n}-gram", "Matched": matched, "In hypothesis": total, "Precision": round(p, 3)})

    if not hyp or min(precisions) == 0:
        return {"bleu": 0.0, "bp": 0.0, "rows": rows, "ref_len": len(ref), "hyp_len": len(hyp)}

    brevity_penalty = 1.0 if len(hyp) > len(ref) else math.exp(1 - len(ref) / len(hyp))
    geo_mean = math.exp(sum(math.log(p) for p in precisions) / max_n)
    return {"bleu": 100 * brevity_penalty * geo_mean, "bp": brevity_penalty, "rows": rows,
            "ref_len": len(ref), "hyp_len": len(hyp)}


# ----------------------------------------------------------------------------------------
# 1. Benchmark
# ----------------------------------------------------------------------------------------
LANGUAGE_PAIRS = [
    ("English - German", "both directions", "Yes (EN-DE)"),
    ("English - French", "both directions", "Yes (EN-FR)"),
    ("English - Czech", "both directions", "No"),
    ("English - Russian", "both directions", "No"),
    ("English - Hindi", "both directions (experimental, low-resource)", "No"),
]


def _render_benchmark() -> None:
    st.subheader("What benchmark was used to evaluate the Transformer?")
    st.markdown(
        "**WMT** stands for *Workshop on Statistical Machine Translation*. "
        "It is a **shared task**: every year, research groups translate the same unseen news texts, and the organisers score "
        "all submissions in the same way. The paper uses the **WMT 2014** edition, "
        "which was the standard benchmark for machine translation at the time the paper was released."
    )

    st.markdown("#### How a shared task works?")
    steps = [
        ("Step 1", "Organisers release", "Training data and a Development set"),
        ("Step 2", "Teams build systems", "Train on the data, tune on the Dev set"),
        ("Step 3", "Teams translation", "A hidden test set (newstest2014)"),
        ("Step 4", "Organisers score", "Human ranking and automatic metrics"),
    ]
    for col, (label, title, body) in zip(st.columns(4), steps):
        with col:
            with st.container(border=True):
                st.caption(label)
                st.markdown(f"**{title}**")
                st.markdown(body)

    st.markdown("#### The five language pairs of WMT 2014")
    md_table(["Language pair", "Translation directions", "Used in the paper?"], LANGUAGE_PAIRS)
    st.caption(
        "Source: WMT 2014 translation task page. The paper uses only EN-DE and EN-FR."
    )

    st.markdown("#### Why a benchmark at all?")
    c1, c2, c3 = st.columns(3)
    with c1:
        with st.container(border=True):
            st.markdown("**Comparable systems**")
            st.markdown("Every system mentioned in the paper is scored on the *same* test set (newstest2014), so the table compares them side-by-side.")
    with c2:
        with st.container(border=True):
            st.markdown("**Unseen set**")
            st.markdown("The Test set is held back from participants training and tuning phases, to avoid data leakage and ensure a high score cannot come from memorising it.")
    with c3:
        with st.container(border=True):
            st.markdown("**Public sets**")
            st.markdown("Training data, Dev sets and Test sets are downloadable, so anyone can re-run and check a claim.")

    # speaker_notes(
    #     "- WMT = a yearly translation competition. Everybody translates the same hidden news sentences; the organisers score everyone the same way.\n"
    #     "- 2014 had five language pairs. The paper uses two: English-German and English-French.\n"
    #     "- Why it matters: every system in Table 2 reports a score on newstest2014, so the comparison is fair.\n"
    #     "- WMT itself used human ranking plus automatic metrics. The paper reports only BLEU: say it plainly.\n"
    #     "- It is a proxy: news only, 3,003 sentences, one reference.\n"
    #     "- Bridge: *\"So what exactly was in the data? Let's look at real sentences.\"*"
    # )


# ----------------------------------------------------------------------------------------
# 2. Data
# ----------------------------------------------------------------------------------------
# Download sizes (GB) of the training corpora, as listed on the WMT 2014 translation task page
CORPUS_SIZES_GB = {
    "Europarl v7": 0.628,
    "Common Crawl": 0.876,
    "News Commentary": 0.077,
    "UN corpus": 2.3,
    "10^9 French-English corpus": 2.3,
}
CORPORA_BY_TASK = {
    "English-German": ["Europarl v7", "Common Crawl", "News Commentary"],
    "English-French": ["Europarl v7", "Common Crawl", "News Commentary", "UN corpus", "10^9 French-English corpus"],
}


def _words(text: str) -> int:
    return len(text.split())


def _render_data() -> None:
    st.subheader("What data was the Transformer trained and tested on?")
    st.markdown(
        "The authors trained on the **WMT 2014 English-German** and **WMT 2014 English-French** training data and measured quality on the "
        "official **newstest2014** test sets. Both datasets are collections of **sentence pairs**: the same sentence written in two languages "
        "by a human translator. The model never sees whole documents, only one sentence pair at a time."
    )

    # tabs = st.tabs(["📦 Corpora & Splits", "🔍 Sample Sentences"])
    # with tabs[0]:
    #     _render_corpora()
    # with tabs[1]:
    _render_samples()


def _render_samples() -> None:
    st.subheader("Sample sentence pairs from WMT 2014 datasets")
    st.caption(
        "Extracted from the public Hugging Face dataset `wmt/wmt14`. "
        "Row numbers are positions inside each split."
    )
    split = st.radio("Split", ["Training", "Development (newstest2013)", "Test (newstest2014)"], horizontal=True, key="res_samples_split")

    if split == "Training":
        pair = st.radio("Language pair", ["English-to-German", "English-to-French"], horizontal=True, key="res_samples_pair")
        if pair == "English-to-German":
            rows, target, label = wmt_samples.TRAIN_DE, "de", "German"
        else:
            rows, target, label = wmt_samples.TRAIN_FR, "fr", "French"
        df = pd.DataFrame(
            [(r["row"], r["en"], r[target], f"{_words(r['en'])} / {_words(r[target])}") for r in rows],
            columns=["Row", "English", label, "Words (EN / target)"],
        )
        st.dataframe(df, width="stretch", hide_index=True)
    else:
        rows = wmt_samples.DEV if split.startswith("Development") else wmt_samples.TEST
        df = pd.DataFrame(
            [(r["en"], r["de"], r["fr"], f"{_words(r['en'])} / {_words(r['de'])} / {_words(r['fr'])}") for r in rows],
            columns=["English", "German", "French", "Words (EN / DE / FR)"],
        )
        st.dataframe(df, width="stretch", hide_index=True)
        st.caption(
            "In the dev and test sets the **same English sentence** has a human German translation and a human French translation. "
            "When the Transformer translates English into German, the German column is the *reference* its output is compared with (BLEU)."
        )

    st.markdown("#### What to notice in the samples")
    n1, n2 = st.columns(2)
    with n1:
        with st.container(border=True):
            st.markdown("**One long German word**")
            st.markdown(
                "Dev row 52: *Prostatakrebsfrüherkennung* is a single German word for *prostate cancer screening* (three English words). "
                "German glues words into compounds, one reason the paper needs subword units."
            )
        with st.container(border=True):
            st.markdown("**Different word order**")
            st.markdown(
                "The same dev row: *take the test or not?* becomes *sollte der Test **gemacht werden** oder nicht?*. "
                "The German verb group ends up at the end of the clause, so the decoder cannot translate word by word."
            )
    with n2:
        with st.container(border=True):
            st.markdown("**Names travel unchanged**")
            st.markdown(
                "Dev row 0: *Obama* appears in the English and German sentence, and as *d'Obama* in French, where an apostrophe attaches it to the word before. "
            )
        with st.container(border=True):
            st.markdown("**Sentence by sentence**")
            st.markdown(
                "Every row is one isolated sentence pair. The Transformer is trained and scored sentence by sentence, "
                "with no document context, and BLEU is then pooled over all 3,003 test sentences."
            )

    # speaker_notes(
    #     "- Start with Training: formal European Parliament speeches. Then switch to Test: newspaper sentences. That is the domain shift.\n"
    #     "- Switch to Development and show that the same English sentence has a German and a French reference.\n"
    #     "- Point at row 52: one German word = three English words. This is the motivation for the next tab (subword units).\n"
    #     "- Same row: German word order differs, so translation is not word by word.\n"
    #     "- Row 0: 'Obama' stays 'Obama'. Motivation for the shared vocabulary.\n"
    #     "- Bridge: *\"How does the model turn these strings into numbers? With subword units.\"*"
    # )


# ----------------------------------------------------------------------------------------
# 2. Metrics glossary (BLEU, PPL, FLOPs)
# ----------------------------------------------------------------------------------------
def _render_metrics() -> None:
    st.subheader("Three numbers appear in the results. What does each one measure?")
    glossary = pd.DataFrame(
        [
            ("BLEU", "0-100, higher means better",
             "How many of the machine's 1-, 2-, 3-, 4-word chunks also appear in a human reference translation, with a penalty for too-short output."),
            ("Perplexity (PPL)", "$\\geq$ 1, lower means better",
             "A way to capture the degree of 'uncertainty' a model has in predicting text. It is based on Cross-Entropy Loss of a tokenized sequence"),
            ("FLOPs (training cost)", "lower means cheaper",
             "Floating-point operations needed to train the model: training time $\\times$ number of GPUs $\\times$ assumed sustained GPU speed. An estimate, not a measurement."),
        ],
        columns=["Metric", "Scale", "What it measures"],
    )
    md_table(list(glossary.columns), list(glossary.itertuples(index=False, name=None)))

    st.markdown("#### BLEU in detail")
    st.markdown(
        "**BLEU** (*BiLingual Evaluation Understudy*) is the standard automatic score for translation. "
        "It compares the machine output (*hypothesis*) with one or more human translations (*references*):"
    )
    st.latex(r"\text{BLEU} = \underbrace{\min\!\Big(1,\, e^{\,1 - r/c}\Big)}_{\text{brevity penalty}} \cdot \exp\!\Big(\tfrac14 \sum_{n=1}^{4} \ln p_n\Big)")
    st.markdown(
        "- $p_n$ = **modified n-gram precision**: the share of hypothesis n-grams found in the reference, "
        "where each reference n-gram can only be used up as many times as it occurs there (stops 'the the the the').\n"
        "- The **geometric mean** of $p_1 \\dots p_4$ means a translation must be right at *every* chunk size: "
        "single words (adequacy) and 4-word runs (fluency).\n"
        "- The **brevity penalty** ($c$ = hypothesis length, $r$ = reference length) punishes dropping words, because precision alone rewards short output."
    )

    st.markdown("##### Try it: a sentence-level BLEU calculator")
    ref = st.text_input("Human reference", value="Die Katze saß auf der Matte", key="res_bleu_ref")
    hyp = st.text_input("Machine hypothesis", value="Die Katze sitzt auf der Matte", key="res_bleu_hyp")
    result = toy_bleu(ref, hyp)

    left, right = st.columns([3, 2])
    with left:
        st.dataframe(pd.DataFrame(result["rows"]), width="stretch", hide_index=True)
    with right:
        st.metric("Toy BLEU (0-100)", f"{result['bleu']:.1f}")
        st.metric("Brevity penalty", f"{result['bp']:.2f}", help="1.00 = hypothesis is not shorter than the reference")
        st.caption(f"Reference: {result['ref_len']} tokens | hypothesis: {result['hyp_len']} tokens")
        if result["hyp_len"] < 4:
            st.caption("A hypothesis under 4 tokens has no 4-grams, so sentence-level BLEU is 0. This is one reason real BLEU is computed over a whole corpus.")
    st.caption(
        "Try shortening the hypothesis (brevity penalty drops), repeating a word (modified precision stops the credit), "
        "or reordering words (unigrams still match, 3- and 4-grams collapse). "
        "This is a simplified, smoothed *sentence-level* BLEU for intuition. The paper's numbers are corpus-level over the whole test set."
    )

# ----------------------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------------------
def render_experimental_setup():
    st.header("Experimental Setup: how the results were produced")
    st.caption("Datasets, metrics (BLEU, perplexity, FLOPs) and the training recipe behind every number that follows.")

    tabs = st.tabs(["📔 Benchmark", "🗂️ Data", "📏 Metrics: BLEU & Co."])
    for tab, render in zip(tabs, [_render_benchmark, _render_data, _render_metrics]):
        with tab:
            render()
