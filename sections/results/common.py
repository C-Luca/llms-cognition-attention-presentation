"""
========================================================================================
Module: Results - shared UI helpers
Author: Gusev Roman (Results Interpreter & Challenger)
========================================================================================
"""

import streamlit as st


def speaker_notes(text: str) -> None:
    """Collapsible talking points, same pattern as the Problem Framing section."""
    with st.expander("🎤 Speaker notes"):
        st.markdown(text)


def md_table(columns: list[str], rows: list[tuple]) -> None:
    """Render a table as markdown so LaTeX ($...$) works in the cells; st.dataframe cannot render it."""
    def cell(value) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = ["| " + " | ".join(cell(c) for c in columns) + " |", "|" + "---|" * len(columns)]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in rows]
    st.markdown("\n".join(lines))


def challenger_box(text: str) -> None:
    """A highlighted 'challenger' remark: where the paper's claim needs a caveat."""
    st.warning("**Challenger's note.** " + text)
