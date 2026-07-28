"""Simple public demo for the chronic-pain-nlp package.

Run from the repository root with:
    streamlit run streamlit_app.py

The app processes pasted text in memory and does not intentionally persist inputs.
"""
from __future__ import annotations

import html
import re
from typing import Any

import streamlit as st

from chronic_pain_nlp import analyze_note, analyze_pre_post


st.set_page_config(
    page_title="Chronic Pain NLP Demo",
    page_icon="🩺",
    layout="wide",
)

st.title("Chronic Pain NLP")
st.caption("A small research demo for extracting pain-related concepts from synthetic or approved text.")

st.warning(
    "Research software only—not for diagnosis or treatment. Do not paste protected health "
    "information or other sensitive data into a public deployment."
)

with st.sidebar:
    st.header("Settings")
    mode = st.radio("Analysis mode", ["Single note", "Pre/post comparison"])
    region_of_surgery = st.selectbox(
        "Region of surgery",
        options=["Not specified", "LUMBAR", "CERVICAL"],
        help="Used by surgery-anchored progression and decision rules.",
    )
    enable_trace = st.checkbox(
        "Include rule trace",
        value=False,
        help="Adds detailed extraction and decision metadata.",
    )
    general_pain_metric = st.selectbox(
        "General pain summary",
        options=["max", "mean"],
        disabled=mode == "Single note",
    )

region = None if region_of_surgery == "Not specified" else region_of_surgery


HIGHLIGHT_STYLES = {
    "region": ("Region", "#dbeafe", "#1e3a8a"),
    "general_score": ("General pain score", "#fef3c7", "#78350f"),
    "regional_score": ("Regional pain score", "#fed7aa", "#7c2d12"),
    "neuropathic": ("Neuropathic term", "#f3e8ff", "#581c87"),
    "laterality": ("Laterality", "#dcfce7", "#14532d"),
}


def _term_spans(text: str, terms: list[str], category: str, priority: int) -> list[dict[str, Any]]:
    spans: list[dict[str, Any]] = []
    for term in sorted({str(t).strip() for t in terms if str(t).strip()}, key=len, reverse=True):
        pattern = rf"(?<!\w){re.escape(term)}(?!\w)"
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            spans.append({
                "start": match.start(),
                "end": match.end(),
                "category": category,
                "priority": priority,
            })
    return spans


def _score_spans(text: str, scores: list[dict[str, Any]], category: str, priority: int) -> list[dict[str, Any]]:
    """Highlight the numeric score inside each extractor-provided match span."""
    spans: list[dict[str, Any]] = []
    number_re = re.compile(r"\b\d+(?:\.\d+)?(?:\s*[-–]\s*\d+(?:\.\d+)?)?\b")
    for item in scores or []:
        try:
            start = max(0, int(item["start"]))
            end = min(len(text), int(item["end"]))
        except (KeyError, TypeError, ValueError):
            continue
        if start >= end:
            continue
        snippet = text[start:end]
        match = number_re.search(snippet)
        if match:
            spans.append({
                "start": start + match.start(),
                "end": start + match.end(),
                "category": category,
                "priority": priority,
            })
    return spans


def _laterality_terms(result) -> list[str]:
    aliases = {"L": ["left"], "R": ["right"], "B": ["bilateral", "bilaterally"]}
    terms: set[str] = set()
    for item in result.lateralized_regions or []:
        terms.update(aliases.get(str(item.get("laterality", "")), []))
    return sorted(terms)


def _resolve_overlaps(spans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep higher-priority spans and discard lower-priority overlaps."""
    valid = [s for s in spans if s["start"] < s["end"]]
    ranked = sorted(valid, key=lambda s: (-s["priority"], -(s["end"] - s["start"]), s["start"]))
    kept: list[dict[str, Any]] = []
    for candidate in ranked:
        if any(candidate["start"] < item["end"] and item["start"] < candidate["end"] for item in kept):
            continue
        kept.append(candidate)
    return sorted(kept, key=lambda s: s["start"])


def render_highlighted_note(text: str, result) -> None:
    spans: list[dict[str, Any]] = []
    spans.extend(_score_spans(text, result.regional_scores, "regional_score", 50))
    spans.extend(_score_spans(text, result.general_scores, "general_score", 45))
    spans.extend(_term_spans(text, list(result.neuropathic_hits or []), "neuropathic", 40))
    spans.extend(_term_spans(text, _laterality_terms(result), "laterality", 30))
    spans.extend(_term_spans(text, list(result.regions or []), "region", 20))
    spans = _resolve_overlaps(spans)

    st.subheader("Highlighted note")
    legend = "".join(
        f'<span class="legend-item"><span class="legend-swatch" style="background:{bg}"></span>{html.escape(label)}</span>'
        for label, bg, _ in HIGHLIGHT_STYLES.values()
    )

    pieces: list[str] = []
    cursor = 0
    for span in spans:
        pieces.append(html.escape(text[cursor:span["start"]]))
        label, bg, fg = HIGHLIGHT_STYLES[span["category"]]
        marked = html.escape(text[span["start"]:span["end"]])
        pieces.append(
            f'<mark title="{html.escape(label)}" style="background:{bg};color:{fg};'
            'padding:0.08rem 0.18rem;border-radius:0.2rem;font-weight:600;">'
            f"{marked}</mark>"
        )
        cursor = span["end"]
    pieces.append(html.escape(text[cursor:]))

    st.markdown(
        """
        <style>
        .highlight-legend {display:flex;flex-wrap:wrap;gap:.8rem;margin-bottom:.65rem;font-size:.88rem}
        .legend-item {display:inline-flex;align-items:center;gap:.3rem}
        .legend-swatch {width:.8rem;height:.8rem;border-radius:.18rem;border:1px solid rgba(0,0,0,.15)}
        .highlight-box {white-space:pre-wrap;line-height:1.75;padding:1rem;border:1px solid rgba(128,128,128,.25);
                        border-radius:.5rem;background:rgba(128,128,128,.04);overflow-wrap:anywhere}
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(f'<div class="highlight-legend">{legend}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="highlight-box">{"".join(pieces)}</div>', unsafe_allow_html=True)
    if not spans:
        st.caption("No highlightable extracted spans were found in this note.")


def show_extraction(result, note_text: str | None = None) -> None:
    """Render a PainExtraction result and optional highlighted source text."""
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Regions", len(result.regions))
    c2.metric("General scores", len(result.general_scores))
    c3.metric("Regional scores", len(result.regional_scores))
    c4.metric("Neuropathic language", "Yes" if result.neuropathic_flag else "No")

    if note_text is not None:
        render_highlighted_note(note_text, result)

    st.subheader("Extracted concepts")
    st.json(
        {
            "regions": result.regions,
            "general_scores": result.general_scores,
            "regional_scores": result.regional_scores,
            "neuropathic_flag": result.neuropathic_flag,
            "neuropathic_hits": result.neuropathic_hits,
            "suppress_flag": result.suppress_flag,
            "improvement_flag": result.improvement_flag,
            "lateralized_regions": result.lateralized_regions,
            "laterality_map": result.laterality_map,
            "neuropathic_region_progression": {
                "flag": result.neuropathic_region_progression.flag,
                "details": result.neuropathic_region_progression.details,
            },
        }
    )

    if enable_trace and result.trace is not None:
        with st.expander("Extraction trace"):
            st.json(result.trace)


if mode == "Single note":
    note = st.text_area(
        "Note text",
        height=240,
        placeholder="Synthetic example: left leg pain rated 8/10 with burning discomfort.",
    )

    if st.button("Analyze note", type="primary", use_container_width=True):
        if not note.strip():
            st.error("Enter note text before running the analysis.")
        else:
            try:
                result = analyze_note(
                    note,
                    region_of_surgery=region,
                    enable_trace=enable_trace,
                )
                show_extraction(result, note)
            except Exception as exc:
                st.error(f"Analysis failed: {exc}")
else:
    left, right = st.columns(2)
    with left:
        preop_text = st.text_area(
            "Preoperative note",
            height=240,
            placeholder="Synthetic example: mild low back pain rated 3/10.",
        )
    with right:
        postop_text = st.text_area(
            "Postoperative note",
            height=240,
            placeholder="Synthetic example: left leg pain rated 8/10 with burning discomfort.",
        )

    if st.button("Compare notes", type="primary", use_container_width=True):
        if not preop_text.strip() or not postop_text.strip():
            st.error("Enter both preoperative and postoperative note text.")
        else:
            try:
                result = analyze_pre_post(
                    preop_text,
                    postop_text,
                    region_of_surgery=region,
                    general_pain_metric=general_pain_metric,
                    enable_trace=enable_trace,
                )

                st.subheader("Decision")
                c1, c2 = st.columns(2)
                c1.metric(
                    "Predicted chronic pain",
                    "Positive" if result.predicted_chronic_pain else "Negative",
                )
                c2.metric("Rules triggered", sum(bool(v) for v in result.flags.values()))
                st.json(result.flags)

                pre_tab, post_tab = st.tabs(["Preoperative extraction", "Postoperative extraction"])
                with pre_tab:
                    show_extraction(result.preop, preop_text)
                with post_tab:
                    show_extraction(result.postop, postop_text)

                if enable_trace and result.decision_trace:
                    with st.expander("Decision trace"):
                        st.json(result.decision_trace)
            except Exception as exc:
                st.error(f"Analysis failed: {exc}")

st.divider()
st.caption("Inputs are processed in memory by this app. Hosting platforms may have their own logging policies.")
