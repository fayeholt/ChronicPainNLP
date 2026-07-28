#!/usr/bin/env python3
"""
extractor.py — Pain extraction utilities (single-file version)
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union
import os
import pandas as pd


try:
    import yaml  # type: ignore
except Exception as e:  # pragma: no cover
    raise ImportError("PyYAML is required to load pain_rules.yaml. Install with `pip install pyyaml`.") from e

# Resolve YAML path: env override first, then module-relative <package>/rules/pain_rules.yaml
PACKAGE_DIR = Path(__file__).resolve().parent
_RULES_PATH = os.environ.get("PAIN_RULES_PATH") or str(PACKAGE_DIR / "rules" / "pain_rules.yaml")
_RULES_PATH = Path(_RULES_PATH).resolve()

if not _RULES_PATH.exists():
    raise FileNotFoundError(
        f"pain_rules.yaml not found at {_RULES_PATH}. "
        "Set PAIN_RULES_PATH to your YAML file or place it at <package>/rules/pain_rules.yaml."
    )

with _RULES_PATH.open("r", encoding="utf-8") as _f:
    _RULES = yaml.safe_load(_f)

if not isinstance(_RULES, dict):
    raise ValueError(f"Top-level of { _RULES_PATH } must be a YAML mapping (dict).")

# Normalize regions.composite_labels keys if provided as strings like "(CERVICAL, neck)"
def _normalize_composite_labels_in_rules() -> None:
    comp = _RULES.get("regions", {}).get("composite_labels")
    if not isinstance(comp, dict):
        return
    normalized: Dict[Tuple[str, str], str] = {}
    for k, v in comp.items():
        if isinstance(k, str) and k.startswith("(") and k.endswith(")"):
            parts = k[1:-1].split(",")
            if len(parts) == 2:
                normalized[(parts[0].strip(), parts[1].strip())] = v
        elif isinstance(k, (list, tuple)) and len(k) == 2:
            normalized[(str(k[0]).strip(), str(k[1]).strip())] = v
        else:
            normalized[k] = v
    _RULES["regions"]["composite_labels"] = normalized

_normalize_composite_labels_in_rules()

# Strict dotted-path accessor: ignores `default` and raises if missing/null
def _rules(path: str, default: Any = None) -> Any:
    cur: Any = _RULES
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            raise KeyError(f"Missing required config key '{path}' in { _RULES_PATH }")
        cur = cur[part]
    if cur is None:
        raise KeyError(f"Config key '{path}' is present but null in { _RULES_PATH }")
    return cur

# -----------------------------------------------------------------------------
# Sentence splitting
# -----------------------------------------------------------------------------
SENTENCE_SPLIT_PATTERN: str = _rules("sentence.split_pattern")
SENTENCE_SPLIT = re.compile(SENTENCE_SPLIT_PATTERN)

# -----------------------------------------------------------------------------
# Negation & persistence
# -----------------------------------------------------------------------------
NEGATION_CUES_REGEX = re.compile(
    "|".join(_RULES["negation"]["cues"]), flags=re.IGNORECASE
)

PREOP_SYMPTOM_IGNORE = re.compile(
    _rules("negation.preop_symptom_ignore_pattern"),
    flags=re.IGNORECASE,
)

PERSISTENCE_CUES: List[str] = _rules("neuropathy.persistence_cues")

# -----------------------------------------------------------------------------
# Date guard and denominator regexes
# -----------------------------------------------------------------------------
DATE_PATTERNS: List[str] = _rules("date_guard.patterns")
_DATE_RE = re.compile("|".join(DATE_PATTERNS), flags=re.IGNORECASE)

_DENOM_RE = re.compile(_rules("pain_scores.denom_regex"), flags=re.IGNORECASE)
_RANGE_DENOM_RE = re.compile(_rules("pain_scores.range_denom_regex"), flags=re.IGNORECASE)

# -----------------------------------------------------------------------------
# Truncation
# -----------------------------------------------------------------------------
CUTOFF_PATTERNS: List[str] = _rules("truncate.cutoff_patterns")

# -----------------------------------------------------------------------------
# Regions (variants, composites, back guards)
# -----------------------------------------------------------------------------
PAIN_REGION_VARIANTS: Dict[str, List[str]] = _rules("regions.variants")
PAIN_REGIONS: List[str] = sorted({v.lower() for vs in PAIN_REGION_VARIANTS.values() for v in vs})

COMPOSITE_REGION_LABELS: Dict[Tuple[str, str], str] = _rules("regions.composite_labels")

_BACK_FALSE_POSITIVE_FOLLOW = re.compile(_rules("regions.back_false_positive_follow"), flags=re.IGNORECASE)
_BACK_FALSE_POSITIVE_VERBS  = re.compile(_rules("regions.back_false_positive_verbs"),  flags=re.IGNORECASE)
_BACK_ANATOMICAL_CUES       = re.compile(_rules("regions.back_anatomical_cues"),       flags=re.IGNORECASE)

# -----------------------------------------------------------------------------
# Region progression chains
# -----------------------------------------------------------------------------
LUMBAR_CHAIN: List[str]   = _rules("progression.lumbar_chain")
CERVICAL_CHAIN: List[str] = _rules("progression.cervical_chain")
SURGERY_TO_CHAINS: Dict[str, List[List[str]]] = _rules("progression.surgery_to_chains")

# -----------------------------------------------------------------------------
# Laterality
# -----------------------------------------------------------------------------
LATERALITY_ALIASES: Dict[str, List[str]] = _rules("laterality.aliases")
LATERALIZABLE_ATOMIC_REGIONS: Set[str] = set(_rules("laterality.lateralizable_atomic_regions"))
_LATERALITY_WINDOW: int = int(_rules("laterality.window_chars"))

# -----------------------------------------------------------------------------
# Pain score patterns (general + regional)
# -----------------------------------------------------------------------------
# If you rely on code-assembled tokens like SCORE_NUM_BARE, keep them defined here.
SCORE_NUM_BARE   = r"(\d+(?:\.\d+)?)(?!\s*[/-]\s*\d)"
SCORE_NUM_DENOM  = r"(\d+(?:\.\d+)?)"
SCORE_RANGE      = r"(\d+(?:\.\d+)?\s*[-–]\s*\d+(?:\.\d+)?)(?!\s*-\s*\d)"
PAIN_SCORE_VERB_PATTERN = (
    r"(?:is|was|were|rated|scored|rating|scoring|"
    r"reported|reports|described|describes|"
    r"noted|documented|measured|"
    r"as\s+being)"
)
# Helper used by previous defaults; YAML should provide full patterns now.
DENOM_ANY = rf"\b({SCORE_RANGE}|{SCORE_NUM_DENOM})\s*(?:/|out\s+of\s*)\s*10(?:\.0+)?\b"

GENERAL_PAIN_PATTERNS: List[str]  = _rules("pain_scores.general_patterns")
REGIONAL_PAIN_PATTERNS: List[str] = _rules("pain_scores.regional_patterns")

# -----------------------------------------------------------------------------
# Neuropathy patterns
# -----------------------------------------------------------------------------
NEUROPATHIC_KEYWORDS: List[str] = _rules("neuropathy.keywords")
NEUROPATHIC_PATTERN = re.compile(r"\b(" + "|".join(NEUROPATHIC_KEYWORDS) + r")\b", flags=re.IGNORECASE)

_NEURO_INCISION_RE = re.compile(_rules("neuropathy.incision_regex"), flags=re.IGNORECASE)

# -----------------------------------------------------------------------------
# Suppression language
# -----------------------------------------------------------------------------
ABSENCE_VERBS: List[str]     = _rules("suppression.absence_verbs")
STABILITY_VERBS: List[str]   = _rules("suppression.stability_verbs")
NEWNESS_NEGATORS: List[str]  = _rules("suppression.newness_negators")
SYMPTOM_TARGETS: List[str]   = _rules("suppression.symptom_targets")

# -----------------------------------------------------------------------------
# Improvement language
# -----------------------------------------------------------------------------
IMPROVEMENT_CUES: List[str]     = _rules("improvement.cues")
IMPROVEMENT_TARGETS: List[str]  = _rules("improvement.targets")
IMPROVEMENT_NEGATORS: List[str] = _rules("improvement.negators")

# =============================================================================
# 1) Sentence utilities (sentence-scoped heuristics)
# =============================================================================

def _sentence_has(patterns: Sequence[str], sentence: str) -> bool:
    return any(re.search(p, sentence) for p in patterns)

def _sentence_bounds(text: str, idx: int) -> tuple[int, int]:
    """
    Return (sent_start, sent_end) around idx using simple punctuation/newline boundaries.
    Boundaries: '.', '\\n', ';'
    """
    if not text:
        return 0, 0

    left_chunk = text[:idx]
    last_dot = left_chunk.rfind(".")
    last_nl = left_chunk.rfind("\n")
    last_sc = left_chunk.rfind(";")

    sent_start = max(last_dot, last_nl, last_sc)
    sent_start = 0 if sent_start == -1 else sent_start + 1  # move past boundary char

    right_dot = text.find(".", idx)
    right_nl = text.find("\n", idx)
    right_sc = text.find(";", idx)
    candidates = [p for p in (right_dot, right_nl, right_sc) if p != -1]
    sent_end = min(candidates) if candidates else len(text)

    return sent_start, sent_end

def _iter_sentences_with_spans(text: str):
    """
    Yield (sentence, start_offset_in_text).
    Uses SENTENCE_SPLIT but keeps offsets so we can map match spans.
    """
    if not text:
        return
    starts = [0]
    for m in SENTENCE_SPLIT.finditer(text):
        starts.append(m.end())
    starts.append(len(text))

    for i in range(len(starts) - 1):
        s0, s1 = starts[i], starts[i + 1]
        sent = text[s0:s1]
        if sent.strip():
            yield sent, s0

# =============================================================================
# Negation via medSpacy
# =============================================================================
try:
    import medspacy  # clinical NLP toolkit built on spaCy
except Exception as e:
    raise ImportError(
        "medspaCy is required for package-based negation. Install with `pip install medspacy`."
    ) from e

# Initialize medspaCy pipeline once
_NLP = medspacy.load()  # loads a spaCy pipeline with medspacy_context included
if "sentencizer" not in _NLP.pipe_names:
    # Make sure sentences are available if not already (some medspaCy configs have segmenters)
    _NLP.add_pipe("sentencizer", first=True)

# ConText component
try:
    _CONTEXT = _NLP.get_pipe("medspacy_context")
except Exception:
    # If not present for some reason, add it
    from medspacy.context import ConTextComponent
    _CONTEXT = ConTextComponent(_NLP)
    _NLP.add_pipe(_CONTEXT, last=True)

# Use preop ignore from YAML (strict accessor)
PREOP_SYMPTOM_IGNORE = re.compile(_rules("negation.preop_symptom_ignore_pattern"), flags=re.IGNORECASE)

def is_negated(text: str, match_start: int, match_end: int | None = None, window: int = 40) -> bool:
    """
    Determine if a matched span is negated using medspaCy ConText
    and additional negation cues loaded from YAML.
    """
    if not text:
        return False

    # Get sentence containing the match
    sent_start, sent_end = _sentence_bounds(text, match_start)
    sentence = text[sent_start:sent_end]


    rel_start = max(0, match_start - sent_start)
    rel_end = (match_end - sent_start) if match_end is not None else (rel_start + 1)

    doc = _NLP(sentence)

    # Align char span to token boundaries
    span = doc.char_span(rel_start, rel_end, alignment_mode="expand")
    if span is None:
        return False

    # Inject span and run ConText
    doc.ents = tuple(list(doc.ents) + [span])
    _CONTEXT(doc)

    # First check medspaCy ConText
    if getattr(span._, "is_negated", False):
        return True

    # Then check YAML-defined negation cues in a window
    window_start = max(0, rel_start - window)
    window_end = min(len(sentence), rel_end + window)
    context_text = sentence[window_start:window_end]

    if NEGATION_CUES_REGEX.search(context_text):
        return True

    # preop ignore
    if PREOP_SYMPTOM_IGNORE.search(sentence):
        return True

    return False

def region_in_negated_symptom_sentence(text: str, region_start: int, region_end: int) -> bool:
    """
    Suppress region mentions like:
        'no weakness in arms'
        'denies pain in leg'
        'no numbness of hand'
    """

    sent_start, sent_end = _sentence_bounds(text, region_start)
    sentence = text[sent_start:sent_end]

    doc = _NLP(sentence)

    for ent in doc.ents:
        # If a clinical concept is negated
        if getattr(ent._, "is_negated", False):
            # If region lies inside or very near this negated concept span
            if ent.start_char <= (region_start - sent_start) <= ent.end_char:
                return True

            # Prepositional attachment pattern: "no weakness in arms"
            # region appears after negated concept inside same sentence
            if ent.end_char <= (region_start - sent_start) <= ent.end_char + 25:
                return True

    return False

# =============================================================================
# Persistence and suppression
# =============================================================================

def has_persistence_cue(text: str, match_start: int, window: int = 40) -> bool:
    start = max(0, match_start - window)
    context = text[start:match_start].lower()
    return any(re.search(p, context) for p in PERSISTENCE_CUES)



def has_suppressing_language(text: str) -> bool:
    """
    True if any sentence indicates:
      1) absence of symptoms, OR
      2) stable/unchanged pain, OR
      3) denial of NEW symptoms
    """
    if not text:
        return False

    for sent in SENTENCE_SPLIT.split(text):
        s = sent.lower()
        if _sentence_has(ABSENCE_VERBS, s) and _sentence_has(SYMPTOM_TARGETS, s):
            return True
        if _sentence_has(STABILITY_VERBS, s) and re.search(r"\bpain\b", s):
            return True
        if _sentence_has(NEWNESS_NEGATORS, s) and _sentence_has(SYMPTOM_TARGETS, s):
            return True

    return False


# =============================================================================
# 3) Date-like guard (used to avoid treating dates as pain scores)
# =============================================================================

def _date_like_near_span(
    text: str,
    span_start: int,
    span_end: int,
    *,
    window: int = 8,
    span_text: Optional[str] = None,
) -> bool:
    """
    True if a date-like string appears near the matched span.

    Key behavior:
    - If the span OR the immediate tail indicates "/10" or "out of 10", do NOT treat as date.
    - Exempt range-denominator forms like "7-9/10" even though they match mm-dd-yy style.
    """
    if not text:
        return False

    # 1) If the matched span itself contains a denominator, it's not a date.
    if span_text and _DENOM_RE.search(span_text):
        return False

    # 2) If a denominator appears immediately after the span, it's not a date.
    #    This fixes cases where the match span is "7-9" and "/10" is outside the span.
    tail = text[span_start : min(len(text), span_end + 16)]
    if _DENOM_RE.search(tail):
        return False

    # 3) Context window
    a = max(0, span_start - window)
    b = min(len(text), span_end + window)
    ctx = text[a:b]

    # 4) Explicitly exempt "range/10" (e.g. "7-9/10") from date detection.
    if _RANGE_DENOM_RE.search(ctx):
        return False

    return bool(_DATE_RE.search(ctx))


# =============================================================================
# 4) Truncation (cut off the note at headers to reduce false positives)
# =============================================================================


def truncate_before_cutoffs(text: str) -> str:
    if not text:
        return text

    text_lower = text.lower()
    cut_positions: List[int] = []
    for pat in CUTOFF_PATTERNS:
        m = re.search(pat, text_lower)
        if m:
            cut_positions.append(m.start())
    return text[: min(cut_positions)] if cut_positions else text


# =============================================================================
# 5) Regions (surface forms + surgery-aware mapping)
# =============================================================================
def _is_anatomical_back(text: str, start: int, end: int) -> bool:
    """
    Decide whether a matched token 'back' is a body-region mention.

    Rules:
    1) If followed immediately by 'of/to/...' => NOT anatomical ("back of", "back to")
    2) If followed by common non-body continuations ("back to work") => NOT anatomical
    3) If the sentence contains strong anatomical cues => anatomical
    4) Else: default to anatomical (keeps recall), but you can flip to False if you want precision.
    """
    sent_start, sent_end = _sentence_bounds(text, start)
    sent = text[sent_start:sent_end]

    # If the *local* right-context looks like "back of/to/..."
    right = text[end : min(len(text), end + 30)]
    if _BACK_FALSE_POSITIVE_FOLLOW.search(right):
        return False
    if _BACK_FALSE_POSITIVE_VERBS.search(right):
        return False

    # Sentence-level anatomical reinforcement
    if _BACK_ANATOMICAL_CUES.search(sent):
        return True

    # Default behavior: keep as anatomical unless it hits a known FP pattern
    return True

def normalize_region_with_surgery(raw_region: str, region_of_surgery: Optional[str]) -> str:
    if not raw_region or not region_of_surgery:
        return raw_region
    return COMPOSITE_REGION_LABELS.get(
        (region_of_surgery.upper(), raw_region.lower()),
        raw_region.lower(),
    )


def canonicalize_region_surface(surface: str) -> str:
    """Map a surface form (e.g. 'feet') to canonical key (e.g. 'foot')."""
    s = (surface or "").lower()
    for canonical, variants in PAIN_REGION_VARIANTS.items():
        if s == canonical or s in {v.lower() for v in variants}:
            return canonical
    return s


def extract_regions(text: str) -> List[str]:
    """
    Extract *surface-form* region mentions (e.g. "feet", "hand") that are NOT negated.
    Returns surface forms, not canonical keys.
    """
    if not text:
        return []

    found: Set[str] = set()

    for region in PAIN_REGIONS:
        for m in re.finditer(rf"\b{re.escape(region)}\b", text, flags=re.IGNORECASE):
            if is_negated(text, m.start(), m.end()):
                continue

            if region_in_negated_symptom_sentence(text, m.start(), m.end()):
                continue

            # Special guard for "back"
            if region.lower() == "back":
                if not _is_anatomical_back(text, m.start(), m.end()):
                    continue

            found.add(region.lower())

    return sorted(found)


# =============================================================================
# 6) Region progression logic (used by decision_tree.py)
# =============================================================================
def progressed(pre_set: Set[str], post_set: Set[str], chain: List[str], bidirectional: bool = False) -> bool:
    """
    Progression within a chain by earliest mentioned region index.
    - bidirectional=False: only downstream
    - bidirectional=True: any change in earliest index
    """
    pre_idxs = [i for i, r in enumerate(chain) if r in pre_set]
    post_idxs = [i for i, r in enumerate(chain) if r in post_set]
    if not pre_idxs or not post_idxs:
        return False
    return (min(post_idxs) != min(pre_idxs)) if bidirectional else (min(post_idxs) > min(pre_idxs))


def compute_region_progression_flag(pre_atomic: Set[str], post_atomic: Set[str], region_of_surgery: Union[str, Set]) -> bool:
    if not pre_atomic or not post_atomic or not region_of_surgery:
        return False

    pre_atomic = {canonicalize_region_surface(r) for r in pre_atomic}
    post_atomic = {canonicalize_region_surface(r) for r in post_atomic}

    surgery_regions = {region_of_surgery.upper()} if isinstance(region_of_surgery, str) else {str(r).upper() for r in region_of_surgery}
    for surg in surgery_regions:
        for chain in SURGERY_TO_CHAINS.get(surg, []):
            if progressed(pre_atomic, post_atomic, chain):
                return True
            if chain == CERVICAL_CHAIN and progressed(post_atomic, pre_atomic, chain):
                return True
    return False


def compute_region_progression_details(pre_atomic: Set[str], post_atomic: Set[str], region_of_surgery: Union[str, Set]):
    if not pre_atomic or not post_atomic or not region_of_surgery:
        return False, None

    pre_atomic = {canonicalize_region_surface(r) for r in pre_atomic}
    post_atomic = {canonicalize_region_surface(r) for r in post_atomic}

    surgery_regions = {region_of_surgery.upper()} if isinstance(region_of_surgery, str) else {str(r).upper() for r in region_of_surgery}

    for surg in surgery_regions:
        for chain in SURGERY_TO_CHAINS.get(surg, []):
            pre_idxs = [i for i, r in enumerate(chain) if r in pre_atomic]
            post_idxs = [i for i, r in enumerate(chain) if r in post_atomic]
            if not pre_idxs or not post_idxs:
                continue

            if min(post_idxs) > min(pre_idxs):
                return True, {
                    "surgery": surg,
                    "chain": chain,
                    "pre_region": chain[min(pre_idxs)],
                    "post_region": chain[min(post_idxs)],
                    "direction": "downstream",
                }

            if chain == CERVICAL_CHAIN and min(pre_idxs) > min(post_idxs):
                return True, {
                    "surgery": surg,
                    "chain": chain,
                    "pre_region": chain[min(pre_idxs)],
                    "post_region": chain[min(post_idxs)],
                    "direction": "bidirectional",
                }

    return False, None


# =============================================================================
# 7) Laterality (L/R/B around region mentions)
# =============================================================================

def _detect_laterality_near(sentence: str, span_start: int, span_end: int, window_chars: int = 20) -> str:
    """
    Require laterality to appear BEFORE the region mention.
    Looks only in the left window [span_start - window_chars, span_start).
    """
    left = max(0, span_start - window_chars)
    # only BEFORE the region
    ctx = sentence[left:span_start].lower()

    # Bilateral first
    for pat in LATERALITY_ALIASES.get("B", []):
        if re.search(pat, ctx, flags=re.IGNORECASE):
            return "B"
    for pat in LATERALITY_ALIASES.get("L", []):
        if re.search(pat, ctx, flags=re.IGNORECASE):
            return "L"
    for pat in LATERALITY_ALIASES.get("R", []):
        if re.search(pat, ctx, flags=re.IGNORECASE):
            return "R"
    return "U"


def extract_lateralized_regions(text: str) -> List[Dict[str, Any]]:
    """
    Extract laterality mentions for a subset of regions:
      {"atomic_region": "hand", "laterality": "L", "text": "...", "sentence": "..."}
    """
    if not text:
        return []

    hits: List[Dict[str, Any]] = []
    for sent, sent_start in _iter_sentences_with_spans(text):
        for surface in PAIN_REGIONS:
            atomic = canonicalize_region_surface(surface)
            if atomic not in LATERALIZABLE_ATOMIC_REGIONS:
                continue

            for m in re.finditer(rf"\b{re.escape(surface)}\b", sent, flags=re.IGNORECASE):
                abs_start = sent_start + m.start()
                if is_negated(text, abs_start):
                    continue

                lat = _detect_laterality_near(sent, m.start(), m.end(), window_chars=20)
                if lat == "U":
                    continue

                snippet = sent[max(0, m.start() - 30): min(len(sent), m.end() + 30)].strip()
                hits.append(
                    {
                        "atomic_region": atomic,
                        "laterality": lat,
                        "text": snippet,
                        "sentence": sent.strip(),
                        "start": abs_start,
                    }
                )

    # de-dupe
    seen = set()
    out: List[Dict[str, Any]] = []
    for h in hits:
        key = (h["atomic_region"], h["laterality"], h["sentence"])
        if key not in seen:
            seen.add(key)
            out.append(h)
    return out


def extract_laterality_map(text: str) -> Dict[str, List[str]]:
    """
    Aggregate laterality signals into {atomic_region: ["L","R",...]}.
    Bilateral ("B") implies both sides (adds "L" and "R").
    """
    if not text:
        return {}

    out: Dict[str, Set[str]] = {}

    for sent in SENTENCE_SPLIT.split(text):
        s_lower = sent.lower()
        for canonical, variants in PAIN_REGION_VARIANTS.items():
            if canonical not in LATERALIZABLE_ATOMIC_REGIONS:
                continue
            for v in variants:
                for m in re.finditer(rf"\b{re.escape(v.lower())}\b", s_lower):
                    lat = _detect_laterality_near(sent, m.start(), m.end(), window_chars=20)
                    if lat != "U":
                        out.setdefault(canonical, set()).add(lat)

    normalized: Dict[str, List[str]] = {}
    for reg, lats in out.items():
        if "B" in lats:
            lats = set(lats) | {"L", "R"}
        normalized[reg] = sorted(lats)
    return normalized


# =============================================================================
# 8) Pain score extraction (general + explicit regional)
# =============================================================================

def _make_score_item(
    *,
    score: float,
    snippet: str,
    start: int,
    end: int,
    text: str,
    source: str,
) -> Dict[str, Any]:
    """
    Create a normalized score item that always carries provenance + negation.
    """
    return {
        "score": float(score),
        "text": snippet,
        "start": int(start),
        "end": int(end),
        "negated": bool(is_negated(text, start)),
        "source": source,
    }


def extract_general_pain_scores(
    text: str,
    window_chars: int = 50,
    region_of_surgery: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Extract:
      - general_scores: [{"score", "text", "start", "end", "negated", "source"}]
      - regional_scores_from_general (hints): same schema + region_hint fields

    Region hint rule (fixed):
      - Hints are sentence-scoped.
      - A hint is allowed if, in the SAME sentence:
          (a) there is a "pain" anchor near the score mention, AND
          (b) a region surface form occurs near that pain anchor.
      - This prevents cross-sentence leakage:
          "pain in arm. Pain is 9/10"  -> second sentence has no region, so no hint.
      - This still captures:
          "neck pain 5/10" (region before pain, same sentence).

    Hard rule:
      - negated candidates are never emitted.
    """
    if not text:
        return [], []

    general_scores: List[Dict[str, Any]] = []
    hinted_regional_scores: List[Dict[str, Any]] = []

    # how far from the score to look for a pain anchor, inside the same sentence
    PAIN_ANCHOR_WINDOW = max(30, int(window_chars))  # chars
    # how far from the pain anchor to look for a region, inside the same sentence
    REGION_NEAR_PAIN_WINDOW = 60  # chars (tunable)

    for pattern in GENERAL_PAIN_PATTERNS:
        for m in re.finditer(pattern, text, flags=re.IGNORECASE):
            snippet = m.group(0)
            start = m.start()
            end = m.end()

            if _date_like_near_span(text, start, end, window=10, span_text=snippet):
                continue
            if is_negated(text, start):
                continue

            raw = (m.group(1) or "").strip()
            if not raw:
                continue

            # Parse score (range -> max endpoint)
            try:
                if "-" in raw or "–" in raw:
                    parts = re.split(r"[-–]", raw)
                    nums = [
                        float(p.strip())
                        for p in parts
                        if re.match(r"^\d+(\.\d+)?$", p.strip())
                    ]
                    if not nums:
                        continue
                    score = max(nums)
                else:
                    score = float(raw)
            except Exception:
                continue

            if score > 10 or score < 0:
                continue

            # If score is far from a "pain" token, require explicit denominator (same as before)
            start_idx = max(start - window_chars, 0)
            backward_text = text[start_idx:start].lower()
            words = re.findall(r"\w+", backward_text)
            try:
                pain_index = max(i for i, w in enumerate(words) if w == "pain")
                words_between = len(words) - pain_index - 1
            except ValueError:
                words_between = 100

            denominator_present = bool(_DENOM_RE.search(snippet))
            if words_between > 2 and not denominator_present:
                continue

            # -----------------------------
            # SENTENCE-SCOPED REGION HINTING
            # -----------------------------
            sent_start, sent_end = _sentence_bounds(text, start)
            sentence = text[sent_start:sent_end]
            sentence_lower = sentence.lower()

            rel_start = start - sent_start
            rel_end = end - sent_start

            # Find a "pain" anchor near the score mention inside the sentence
            pain_anchor_pos: Optional[int] = None

            # search for "pain" in a window around the score start
            a = max(0, rel_start - PAIN_ANCHOR_WINDOW)
            b = min(len(sentence_lower), rel_start + PAIN_ANCHOR_WINDOW)
            local = sentence_lower[a:b]

            # choose the closest "pain" occurrence to rel_start
            best = None
            for pm in re.finditer(r"\bpain\b", local):
                pos = a + pm.start()
                dist = abs(pos - rel_start)
                if best is None or dist < best[0]:
                    best = (dist, pos)
            if best is not None:
                pain_anchor_pos = best[1]

            raw_region: Optional[str] = None
            if pain_anchor_pos is not None:
                # Search for regions near the pain anchor, still within the same sentence
                ra = max(0, pain_anchor_pos - REGION_NEAR_PAIN_WINDOW)
                rb = min(len(sentence_lower), pain_anchor_pos + REGION_NEAR_PAIN_WINDOW)
                region_ctx = sentence_lower[ra:rb]

                # pick the closest region mention to the pain anchor (not just "last in window")
                closest = None  # (distance, region_surface)
                for r in PAIN_REGIONS:
                    for rm in re.finditer(rf"\b{re.escape(r.lower())}\b", region_ctx):
                        pos = ra + rm.start()
                        dist = abs(pos - pain_anchor_pos)
                        if closest is None or dist < closest[0]:
                            closest = (dist, r.lower())
                if closest is not None:
                    raw_region = closest[1]

            item = _make_score_item(
                score=score,
                snippet=snippet,
                start=start,
                end=end,
                text=text,
                source="general",
            )
            if item["negated"]:
                continue

            if raw_region:
                item["atomic_region_hint"] = raw_region
                item["region_hint"] = normalize_region_with_surgery(raw_region, region_of_surgery)
                item["source"] = "general_hint"
                hinted_regional_scores.append(item)
            else:
                general_scores.append(item)

    return general_scores, hinted_regional_scores


def extract_regional_pain_scores(
    text: str,
    general_hint_scores: Optional[List[Dict[str, Any]]] = None,
    region_of_surgery: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Extract explicit regional pain scores + fold in "general hint" scores.

    Output schema:
      {"region","atomic_region","score","text","start","end","negated","source", ...}
    """
    if not text:
        return []

    regional_scores: List[Dict[str, Any]] = []

    # --- A) explicit regional patterns ---
    for pattern in REGIONAL_PAIN_PATTERNS:
        for m in re.finditer(pattern, text, flags=re.IGNORECASE):
            snippet = m.group(0)
            start = m.start()
            end = m.end()

            if _date_like_near_span(text, start, end, window=10, span_text=snippet):
                continue

            if is_negated(text, start):
                continue

            g = m.groups()
            # For this alternation pattern, groups come in as (reg1, score1, reg2, score2)
            g1, g2, g3, g4 = g[:4]
            raw_region_surface = (g1 or g3 or "").lower()
            score_str = (g2 or g4)

            if not raw_region_surface or score_str is None:
                continue

            # Map surface -> canonical atomic region
            raw_region = None
            for canonical, variants in PAIN_REGION_VARIANTS.items():
                if raw_region_surface == canonical.lower() or raw_region_surface in {v.lower() for v in variants}:
                    raw_region = canonical
                    break
            if not raw_region:
                continue

            try:
                score = int(score_str)
            except Exception:
                continue

            if score > 10 or score < 0:
                continue

            item = {
                "region": normalize_region_with_surgery(raw_region, region_of_surgery),
                "atomic_region": raw_region,
                "score": score,
                "text": snippet,
                "start": start,
                "end": end,
                "negated": bool(is_negated(text, start)),
                "source": "explicit_regional",
            }

            if item["negated"]:
                continue

            # laterality only for lateralizable atomics
            if raw_region in LATERALIZABLE_ATOMIC_REGIONS:
                laterality = "U"
                for sent in SENTENCE_SPLIT.split(text):
                    if snippet.lower() in sent.lower():
                        m2 = re.search(rf"\b{re.escape(raw_region_surface)}\b", sent, flags=re.IGNORECASE)
                        if m2:
                            laterality = _detect_laterality_near(sent, m2.start(), m2.end(), window_chars=20)
                        break
                if laterality != "U":
                    item["laterality"] = laterality

            regional_scores.append(item)

    # --- B) fold in general "region hints" (preserve provenance + negation) ---
    for g in (general_hint_scores or []):
        if not isinstance(g, dict):
            continue
        if g.get("negated") is True:
            continue
        if not g.get("region_hint"):
            continue

        score = g.get("score")
        try:
            score_f = float(score)
        except Exception:
            continue
        if score_f > 10 or score_f < 0:
            continue

        raw_atomic = g.get("atomic_region_hint")
        region = normalize_region_with_surgery(raw_atomic or g["region_hint"], region_of_surgery)

        regional_scores.append(
            {
                "region": region,
                "atomic_region": raw_atomic,
                "score": score_f,
                "text": g.get("text"),
                "start": g.get("start"),
                "end": g.get("end"),
                "negated": bool(g.get("negated", False)),
                "source": "general_hint_folded",
            }
        )

    return regional_scores


# =============================================================================
# 9) Neuropathy features
# =============================================================================

def extract_neuropathic_flag(text: str) -> bool:
    if not text:
        return False
    for m in NEUROPATHIC_PATTERN.finditer(text):
        if is_negated(text, m.start()):
            continue
        if has_persistence_cue(text, m.start()):
            continue
        return True
    return False


def extract_neuropathic_hits(text: str, return_sentences: bool = False) -> List[str] | List[Dict[str, Any]]:
    if not text:
        return []  # type: ignore[return-value]

    hits: List[Any] = []
    sentences = SENTENCE_SPLIT.split(text)

    for m in NEUROPATHIC_PATTERN.finditer(text):
        if is_negated(text, m.start()):
            continue
        if has_persistence_cue(text, m.start()):
            continue

        hit_text = m.group(0).lower()
        if return_sentences:
            sent_for_hit = ""
            for sent in sentences:
                if hit_text in sent.lower():
                    sent_for_hit = sent.strip()
                    break
            hits.append({"hit": hit_text, "sentence": sent_for_hit, "start": m.start()})
        else:
            hits.append(hit_text)

    if return_sentences:
        return hits  # type: ignore[return-value]

    seen: Set[str] = set()
    out: List[str] = []
    for h in hits:
        if h not in seen:
            seen.add(h)
            out.append(h)
    return out


def extract_neuropathic_region_progression(
    text: str,
    *,
    region_of_surgery: Union[str, Set, None],
    preop_regions: Optional[Set[str]] = None,
    window_chars: int = 60,
) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """
    Post-op neuropathic progression:
    - Find a neuropathic keyword (non-negated, not suppressed by persistence)
    - Look +/- window_chars for:
        A) incision/incisional (proxy), OR
        B) region mention in the surgery chain
    Special case: region_of_surgery == "ALL" requires downstream-from-preop logic.
    """
    if not text or not region_of_surgery:
        return False, None

    surgery_regions = (
        {region_of_surgery.upper()}
        if isinstance(region_of_surgery, str)
        else {str(r).upper() for r in region_of_surgery if r}
    )
    is_all = "ALL" in surgery_regions

    surface_re = re.compile(
        r"\b(" + "|".join(map(re.escape, PAIN_REGIONS)) + r")\b",
        flags=re.IGNORECASE,
    )

    preop_canon = {canonicalize_region_surface(v) for v in (preop_regions or set()) if v}

    def _within_chain_relative_to_preop(canonical_region: str) -> Tuple[bool, Dict[str, Any]]:
        if not preop_canon:
            return False, {"reason": "preop_regions_missing_for_ALL"}

        for chain_name, chain in (("LUMBAR_CHAIN", LUMBAR_CHAIN), ("CERVICAL_CHAIN", CERVICAL_CHAIN)):
            if canonical_region not in chain:
                continue

            pre_idxs = [i for i, r in enumerate(chain) if r in preop_canon]
            if not pre_idxs:
                return False, {
                    "reason": "no_preop_region_in_same_chain",
                    "chain_name": chain_name,
                    "preop_canon": sorted(preop_canon),
                }

            post_idx = chain.index(canonical_region)
            pre_min = min(pre_idxs)

            if chain_name == "CERVICAL_CHAIN":
                direction = "same" if post_idx == pre_min else ("downstream" if post_idx > pre_min else "upstream")
                return True, {
                    "chain_name": chain_name,
                    "preop_anchor_region": chain[pre_min],
                    "post_region_index": post_idx,
                    "direction": direction,
                }

            if post_idx >= pre_min:
                return True, {
                    "chain_name": chain_name,
                    "preop_anchor_region": chain[pre_min],
                    "post_region_index": post_idx,
                    "direction": "same" if post_idx == pre_min else "downstream",
                }

            return False, {
                "reason": "upstream_of_preop_not_allowed_for_lumbar",
                "chain_name": chain_name,
                "preop_anchor_region": chain[pre_min],
                "post_region_index": post_idx,
            }

        return False, {"reason": "matched_region_not_in_supported_chains"}

    for m in NEUROPATHIC_PATTERN.finditer(text):
        hit_start = m.start()
        if is_negated(text, hit_start) or has_persistence_cue(text, hit_start):
            continue

        a = max(0, hit_start - window_chars)
        b = min(len(text), m.end() + window_chars)
        ctx = text[a:b]
        ctx_lower = ctx.lower()

        # A) incision proxy
        if re.search(r"\bincision(?:al)?\b", ctx_lower):
            return True, {
                "hit": m.group(0),
                "start": hit_start,
                "surgery_regions": sorted(surgery_regions),
                "matched": "incision_proxy",
                "context": ctx,
                "all_logic_used": is_all,
            }

        # B) closest region mention in window
        best: Optional[Tuple[int, str, str]] = None  # (abs_pos, surface, canonical)
        for rm in surface_re.finditer(ctx):
            surface = rm.group(1)
            canonical = canonicalize_region_surface(surface)
            abs_pos = a + rm.start()
            if best is None or abs(abs_pos - hit_start) < abs(best[0] - hit_start):
                best = (abs_pos, surface, canonical)

        if best is None:
            continue

        _, surface, canonical = best

        if not is_all:
            allowed: Set[str] = set()
            for surg in surgery_regions:
                for chain in SURGERY_TO_CHAINS.get(surg, []):
                    allowed |= set(chain)
            if allowed and canonical in allowed:
                return True, {
                    "hit": m.group(0),
                    "start": hit_start,
                    "surgery_regions": sorted(surgery_regions),
                    "matched": "region_in_surgery_chain_window",
                    "matched_region_surface": surface.lower(),
                    "matched_region_canonical": canonical,
                    "allowed_chain_regions": sorted(allowed),
                    "context": ctx,
                    "all_logic_used": False,
                }
            continue

        # ALL logic requires preop
        if not preop_regions:
            return False, {"reason": "ALL_requires_preop_regions"}

        ok, info = _within_chain_relative_to_preop(canonical)
        if ok:
            return True, {
                "hit": m.group(0),
                "start": hit_start,
                "surgery_regions": sorted(surgery_regions),
                "matched": "all_requires_downstream_from_preop",
                "matched_region_surface": surface.lower(),
                "matched_region_canonical": canonical,
                "preop_regions_canonical": sorted(preop_canon),
                "chain_eval": info,
                "context": ctx,
                "all_logic_used": True,
            }

    return False, None


# =============================================================================
# 10) Improvement language
# =============================================================================

def has_improvement_language(text: str) -> bool:
    if not text:
        return False
    for sent in SENTENCE_SPLIT.split(text):
        s = sent.lower()
        if _sentence_has(IMPROVEMENT_NEGATORS, s):
            continue
        if _sentence_has(IMPROVEMENT_CUES, s) and _sentence_has(IMPROVEMENT_TARGETS, s):
            return True
    return False


def extract_improved_regions(text: str) -> Set[str]:
    if not text:
        return set()

    improved: Set[str] = set()
    for sent in SENTENCE_SPLIT.split(text):
        s = sent.lower()
        if _sentence_has(IMPROVEMENT_NEGATORS, s):
            continue
        if not (_sentence_has(IMPROVEMENT_CUES, s) and _sentence_has(IMPROVEMENT_TARGETS, s)):
            continue

        for canonical, variants in PAIN_REGION_VARIANTS.items():
            for v in variants:
                if re.search(rf"\b{re.escape(v)}\b", s, flags=re.IGNORECASE):
                    improved.add(canonical)
                    break
    return improved


# =============================================================================
# 11) Primary API (what decision_tree.py / Streamlit call)
# =============================================================================

def extract_pain_info(
    text: str,
    region_of_surgery: Optional[str] = None,
    *,
    preop_regions: Optional[Union[Set[str], List[str]]] = None,
) -> Tuple[
    List[str],                 # regions (surface forms)
    List[Dict[str, Any]],       # general_scores
    List[Dict[str, Any]],       # regional_scores
    bool,                       # neuropathic_flag
    List[str],                  # neuropathic_hits
    bool,                       # suppress_flag
    bool,                       # improvement_flag
    List[Dict[str, Any]],       # lateralized_regions
    Dict[str, List[str]],       # laterality_map
    bool,                       # neuropathic_region_progression_flag
    Optional[Dict[str, Any]],   # neuropathic_region_progression_details
]:
    """
    Main extraction entrypoint. This is glue code: calls extractors in fixed order.

    Important invariants:
    - Score dicts include start/end/source/negated.
    - Negated scores are never emitted from score extractors (belt + suspenders).
    - Regional “hints” are folded in exactly once (inside extract_regional_pain_scores).
    """
    # Guard: ensure we have a string
    if text is None or (isinstance(text, float) and pd.isna(text)):
        text = ""
    elif not isinstance(text, str):
        text = str(text)

    # Apply truncation once for the entire pipeline
    text = truncate_before_cutoffs(text)

    regions = extract_regions(text)

    general_scores, regional_scores_from_general_hints = extract_general_pain_scores(
        text,
        region_of_surgery=region_of_surgery,
    )

    # IMPORTANT: pass the HINT items to be folded in (NOT general_scores)
    regional_scores = extract_regional_pain_scores(
        text,
        regional_scores_from_general_hints,
        region_of_surgery=region_of_surgery,
    )

    neuropathic_flag = extract_neuropathic_flag(text)
    neuropathic_hits = extract_neuropathic_hits(text)  # unique list[str]
    suppress_flag = has_suppressing_language(text)
    improvement_flag = has_improvement_language(text)
    lateralized_regions = extract_lateralized_regions(text)
    laterality_map = extract_laterality_map(text)

    preop_set = set(preop_regions) if isinstance(preop_regions, (set, list, tuple)) else None
    neuro_prog_flag, neuro_prog_details = extract_neuropathic_region_progression(
        text,
        region_of_surgery=region_of_surgery,
        window_chars=60,
        preop_regions=preop_set,
    )

    return (
        regions,
        general_scores,
        regional_scores,
        neuropathic_flag,
        neuropathic_hits,
        suppress_flag,
        improvement_flag,
        lateralized_regions,
        laterality_map,
        neuro_prog_flag,
        neuro_prog_details,
    )


# =============================================================================
# 12) DataFrame helpers (stable schema / Streamlit convenience)
# =============================================================================

_PAIN_SCHEMA: Dict[str, Any] = {
    "regions": [],
    "general_pain": [],
    "regional_pain": [],
    "neuropathic_flag": False,
    "neuropathic_hits": [],
    "suppress_flag": False,
    "improvement_flag": False,
    "lateralized_regions": [],
    "laterality_map": {},
    "neuropathic_region_progression_flag": False,
    "neuropathic_region_progression_details": None,
    "pain_trace": None,  # trace-enabled only
}

_PAIN_TUPLE_ORDER_NO_TRACE = [
    "regions",
    "general_pain",
    "regional_pain",
    "neuropathic_flag",
    "neuropathic_hits",
    "suppress_flag",
    "improvement_flag",
    "lateralized_regions",
    "laterality_map",
    "neuropathic_region_progression_flag",
    "neuropathic_region_progression_details",
]

_PAIN_TUPLE_ORDER_WITH_TRACE = [
    "regions",
    "general_pain",
    "regional_pain",
    "neuropathic_flag",
    "neuropathic_hits",
    "suppress_flag",
    "improvement_flag",
    "lateralized_regions",
    "laterality_map",
    "pain_trace",
    "neuropathic_region_progression_flag",
    "neuropathic_region_progression_details",
]


def _coerce_pain_output_to_dict(out, *, with_trace: bool) -> Dict[str, Any]:
    if isinstance(out, dict):
        d = dict(out)
    elif isinstance(out, (tuple, list)):
        keys = _PAIN_TUPLE_ORDER_WITH_TRACE if with_trace else _PAIN_TUPLE_ORDER_NO_TRACE
        d = {k: out[i] for i, k in enumerate(keys) if i < len(out)}
    else:
        d = {}

    base = dict(_PAIN_SCHEMA)
    if not with_trace:
        base.pop("pain_trace", None)
    base.update(d)
    if with_trace and "pain_trace" not in base:
        base["pain_trace"] = None
    return base


def _apply_extractor(df: pd.DataFrame, *, with_surgery: bool, with_trace: bool) -> pd.DataFrame:
    df = df.copy()

    def _call(row: pd.Series) -> Dict[str, Any]:
        txt = row.get("full_note_text", "")
        ros = row.get("region_of_surgery", None) if with_surgery else None
        out = extract_pain_info_with_trace(txt, region_of_surgery=ros) if with_trace else extract_pain_info(txt, region_of_surgery=ros)
        return _coerce_pain_output_to_dict(out, with_trace=with_trace)

    rows = df.apply(_call, axis=1)
    out_df = pd.DataFrame(list(rows), index=df.index)

    expected = list(_PAIN_SCHEMA.keys()) if with_trace else [k for k in _PAIN_SCHEMA.keys() if k != "pain_trace"]
    for c in expected:
        if c not in out_df.columns:
            out_df[c] = _PAIN_SCHEMA[c]

    df[expected] = out_df[expected]
    return df


def extract_df_pain_fields(df: pd.DataFrame) -> pd.DataFrame:
    return _apply_extractor(df, with_surgery=False, with_trace=False)


def extract_df_pain_fields_with_surgery(df: pd.DataFrame) -> pd.DataFrame:
    return _apply_extractor(df, with_surgery=True, with_trace=False)


def extract_df_pain_fields_with_surgery_trace(df: pd.DataFrame) -> pd.DataFrame:
    return _apply_extractor(df, with_surgery=True, with_trace=True)


# =============================================================================
# 13) Debug utilities + trace wrapper (for Streamlit UI)
# =============================================================================

def _find_sentence_for_offset(text: str, abs_start: int) -> str:
    if not text:
        return ""
    for sent, sent_start in _iter_sentences_with_spans(text):
        sent_end = sent_start + len(sent)
        if sent_start <= abs_start < sent_end:
            return sent.strip()
    return ""


def _context_window(text: str, abs_start: int, left: int = 60, right: int = 30) -> str:
    if not text:
        return ""
    a = max(0, abs_start - left)
    b = min(len(text), abs_start + right)
    return text[a:b]


def debug_region_extraction(text: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "cutoff": {"applied": False, "pattern": None, "cut_index": None},
        "mentions": [],
        "final_regions": [],
    }
    if not text:
        return out

    original_text = text
    text_lower = text.lower()
    best: Optional[Tuple[int, str]] = None
    for pat in CUTOFF_PATTERNS:
        m = re.search(pat, text_lower)
        if m and (best is None or m.start() < best[0]):
            best = (m.start(), pat)
    if best is not None:
        out["cutoff"] = {"applied": True, "pattern": best[1], "cut_index": best[0]}
        text = original_text[: best[0]]

    kept: Set[str] = set()
    for region in PAIN_REGIONS:
        for m in re.finditer(rf"\b{re.escape(region)}\b", text, flags=re.IGNORECASE):
            abs_start = m.start()
            neg = is_negated(text, abs_start)
            out["mentions"].append(
                {
                    "surface": region.lower(),
                    "start": abs_start,
                    "match": m.group(0),
                    "negated": neg,
                    "decision": "kept" if not neg else "dropped",
                    "reason": None if not neg else "negated_window",
                    "sentence": _find_sentence_for_offset(text, abs_start),
                    "context": _context_window(text, abs_start),
                }
            )
            if not neg:
                kept.add(region.lower())

    out["final_regions"] = sorted(kept)
    return out


def debug_general_score_extraction(text: str, window_chars: int = 50, region_of_surgery: Optional[str] = None) -> Dict[str, Any]:
    out: Dict[str, Any] = {"matches": [], "final_general": [], "final_hint_regional": []}
    if not text:
        return out

    general_scores, hint_scores = extract_general_pain_scores(text, window_chars=window_chars, region_of_surgery=region_of_surgery)

    for pattern in GENERAL_PAIN_PATTERNS:
        for m in re.finditer(pattern, text, flags=re.IGNORECASE):
            snippet = m.group(0)
            abs_start = m.start()
            abs_end = m.end()

            entry: Dict[str, Any] = {
                "pattern": pattern,
                "start": abs_start,
                "end": abs_end,
                "snippet": snippet,
                "raw_group1": (m.group(1).strip() if m.groups() else None),
                "negated": bool(is_negated(text, abs_start)),
                "date_like": bool(_date_like_near_span(text, abs_start, abs_end, window=10, span_text=snippet)),
                "denominator_present": bool(_DENOM_RE.search(snippet)),
                "decision": None,
                "reason": None,
                "sentence": _find_sentence_for_offset(text, abs_start),
                "context": _context_window(text, abs_start),
            }

            if entry["date_like"]:
                entry["decision"] = "dropped"
                entry["reason"] = "date_like"
            elif entry["negated"]:
                entry["decision"] = "dropped"
                entry["reason"] = "negated_window"
            else:
                entry["decision"] = "candidate"
                entry["reason"] = "passed_basic_filters"

            out["matches"].append(entry)

    out["final_general"] = general_scores
    out["final_hint_regional"] = hint_scores
    return out


def debug_regional_score_extraction(text: str, region_of_surgery: Optional[str] = None) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "matches": [],
        "final_regional": [],
        "folded_from_general": [],   # NEW
    }
    if not text:
        return out

    # 1) generate hint scores the same way the pipeline does
    _, hint_scores = extract_general_pain_scores(text, window_chars=50, region_of_surgery=region_of_surgery)

    # 2) now do regional extraction WITH folding enabled
    regional_scores = extract_regional_pain_scores(
        text,
        general_hint_scores=hint_scores,
        region_of_surgery=region_of_surgery,
    )

    # 3) keep explicit regional regex matches trace
    for pattern in REGIONAL_PAIN_PATTERNS:
        for m in re.finditer(pattern, text, flags=re.IGNORECASE):
            snippet = m.group(0)
            abs_start = m.start()
            abs_end = m.end()
            out["matches"].append(
                {
                    "pattern": pattern,
                    "start": abs_start,
                    "end": abs_end,
                    "snippet": snippet,
                    "negated": bool(is_negated(text, abs_start)),
                    "decision": "kept" if not is_negated(text, abs_start) else "dropped",
                    "reason": None if not is_negated(text, abs_start) else "negated_window",
                    "sentence": _find_sentence_for_offset(text, abs_start),
                    "context": _context_window(text, abs_start),
                }
            )

    # 4) NEW: explicitly show what folding contributed
    out["folded_from_general"] = [
        {
            "region_hint": g.get("region_hint"),
            "atomic_region_hint": g.get("atomic_region_hint"),
            "score": g.get("score"),
            "text": g.get("text"),
            "start": g.get("start"),
            "end": g.get("end"),
            "negated": g.get("negated"),
            "source": g.get("source"),
        }
        for g in (hint_scores or [])
        if isinstance(g, dict)
    ]

    out["final_regional"] = regional_scores
    return out


def debug_neuropathic(text: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {"matches": [], "final_flag": False, "final_hits": []}
    if not text:
        return out

    hits: Set[str] = set()
    for m in NEUROPATHIC_PATTERN.finditer(text):
        abs_start = m.start()
        neg = is_negated(text, abs_start)
        pers = has_persistence_cue(text, abs_start)
        decision = "kept" if (not neg and not pers) else "dropped"
        reason = "negated_window" if neg else ("persistence_cue" if pers else None)

        out["matches"].append(
            {
                "hit": m.group(0),
                "start": abs_start,
                "negated": neg,
                "persistence": pers,
                "decision": decision,
                "reason": reason,
                "sentence": _find_sentence_for_offset(text, abs_start),
                "context": _context_window(text, abs_start),
            }
        )

        if decision == "kept":
            hits.add(m.group(0).lower())

    out["final_hits"] = sorted(hits)
    out["final_flag"] = bool(out["final_hits"])
    return out


def debug_suppressing_language(text: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {"triggered": False, "triggers": []}
    if not text:
        return out

    for sent in SENTENCE_SPLIT.split(text):
        s = sent.lower().strip()
        if not s:
            continue

        if _sentence_has(ABSENCE_VERBS, s) and _sentence_has(SYMPTOM_TARGETS, s):
            out["triggered"] = True
            out["triggers"].append({"rule": "absence+symptom", "sentence": sent.strip()})
            continue
        if _sentence_has(STABILITY_VERBS, s) and re.search(r"\bpain\b", s):
            out["triggered"] = True
            out["triggers"].append({"rule": "stability+pain", "sentence": sent.strip()})
            continue
        if _sentence_has(NEWNESS_NEGATORS, s) and _sentence_has(SYMPTOM_TARGETS, s):
            out["triggered"] = True
            out["triggers"].append({"rule": "no_new+symptom", "sentence": sent.strip()})
            continue

    return out


def debug_laterality(text: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {"mentions": [], "final_map": {}}
    if not text:
        return out

    final_map = extract_laterality_map(text)

    for sent, sent_start in _iter_sentences_with_spans(text):
        for canonical, variants in PAIN_REGION_VARIANTS.items():
            if canonical not in LATERALIZABLE_ATOMIC_REGIONS:
                continue
            for v in variants:
                for m in re.finditer(rf"\b{re.escape(v)}\b", sent, flags=re.IGNORECASE):
                    abs_start = sent_start + m.start()
                    if is_negated(text, abs_start):
                        out["mentions"].append(
                            {
                                "canonical": canonical,
                                "surface": v,
                                "laterality": None,
                                "decision": "dropped",
                                "reason": "negated_window",
                                "sentence": sent.strip(),
                                "context": _context_window(text, abs_start),
                            }
                        )
                        continue

                    lat = _detect_laterality_near(sent, m.start(), m.end(), window_chars=20)
                    if lat == "U":
                        out["mentions"].append(
                            {
                                "canonical": canonical,
                                "surface": v,
                                "laterality": "U",
                                "decision": "dropped",
                                "reason": "no_laterality_nearby",
                                "sentence": sent.strip(),
                                "context": _context_window(text, abs_start),
                            }
                        )
                        continue

                    out["mentions"].append(
                        {
                            "canonical": canonical,
                            "surface": v,
                            "laterality": lat,
                            "decision": "kept",
                            "reason": "laterality_nearby",
                            "sentence": sent.strip(),
                            "context": _context_window(text, abs_start),
                        }
                    )

    out["final_map"] = final_map
    return out


def extract_pain_info_with_trace(text: str, region_of_surgery: Optional[str] = None):
    """
    Same outputs as extract_pain_info, plus a `trace` dict for Streamlit.
    """
    if text is None or (isinstance(text, float) and pd.isna(text)):
        text = ""
    elif not isinstance(text, str):
        text = str(text)

    base_out = extract_pain_info(text, region_of_surgery=region_of_surgery)
    base = _coerce_pain_output_to_dict(base_out, with_trace=False)

    trace = {
        "regions": debug_region_extraction(text),
        "general_scores": debug_general_score_extraction(text, window_chars=50, region_of_surgery=region_of_surgery),
        "regional_scores": debug_regional_score_extraction(text, region_of_surgery=region_of_surgery),
        "neuropathy": debug_neuropathic(text),
        "suppression": debug_suppressing_language(text),
        "laterality": debug_laterality(text),
    }

    return (
        base["regions"],
        base["general_pain"],
        base["regional_pain"],
        base["neuropathic_flag"],
        base["neuropathic_hits"],
        base["suppress_flag"],
        base["improvement_flag"],
        base["lateralized_regions"],
        base["laterality_map"],
        trace,  # pain_trace
        base["neuropathic_region_progression_flag"],
        base["neuropathic_region_progression_details"],
    )