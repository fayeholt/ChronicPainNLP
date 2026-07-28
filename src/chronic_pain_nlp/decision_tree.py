"""Example rule-based decision trees built from ChronicPainNLP features.

This module is the *classification* layer. ``pain_rules.yaml`` configures the
upstream extractor (for example, the phrases that count as pain scores,
anatomic regions, laterality, neuropathic language, negation, or improvement).
The functions in this file decide how those extracted features are combined
into a final ``predicted`` value.

Making your own decision tree
-----------------------------
The safest way to create a site- or study-specific tree is to copy one of the
public functions below, rename it, and change its rules. This keeps the example
tree intact and makes the new phenotype definition easy to review.

1. Define the clinical outcome before changing code.
   Write each criterion in plain language, including its threshold, required
   time point, anatomic restrictions, and whether rules are joined with
   ``AND`` or ``OR``. For example:

       Positive if postoperative general pain is at least 7 OR a pain score in
       the surgical chain increases by at least 2 points from baseline.

2. Choose extractor outputs for each criterion.
   Common inputs include:

   - ``pre_general_scores`` and ``post_general_scores``: dictionaries with a
     numeric ``"score"`` value.
   - ``pre_regional_scores`` and ``post_regional_scores``: dictionaries with
     ``"score"`` plus ``"atomic_region"`` (or ``"region"``).
   - ``pre_regions`` and ``post_regions``: region strings found in the notes.
   - ``region_of_surgery``: a string or set such as ``"LUM_SAC"`` or
     ``{"CERVICAL"}``.
   - neuropathic, improvement, suppression, progression, and laterality
     features produced by the extractor.

   Inspect real extractor output before writing a rule. Do not classify from
   raw note text in this module; add or refine recognition patterns in
   ``pain_rules.yaml`` instead.

3. Add one auditable rule block at a time.
   Each block should calculate a boolean named ``fired``, append a trace entry,
   set a named flag, and optionally return early. A minimal pattern is:

       fired = bool(post_general >= 7)
       if include_trace:
           trace.append({
               "rule": "post_general_at_least_7",
               "post_general": post_general,
               "threshold": 7,
               "fired": fired,
           })
       if fired:
           flags["post_general_at_least_7_flag"] = True
           early = _maybe_return("post_general_at_least_7")
           if early is not None:
               return early

   Add every new flag to the initial ``flags`` dictionary. If the flag should
   make the final prediction positive, also add it to ``positive_flags`` near
   the end of the function. A recorded flag has no effect on ``predicted``
   unless it is included in that final expression (or returned early).

4. Encode combinations explicitly.
   For an ``OR`` tree, include each independently sufficient flag in
   ``positive_flags`` and use ``any(...)``. For an ``AND`` branch, create a
   combined boolean, such as:

       high_and_neuropathic = (
           flags["general_flag"] and flags["neuropathic_flag"]
       )

   For more complex trees, calculate named branch booleans and derive the final
   result from them. Prefer named conditions over a single long expression so
   the trace remains interpretable.

5. Decide how special findings affect the result.
   In this example, some values are retained for interpretation but do not
   independently make the prediction positive. In particular,
   ``new_non_surgical_region_flag``, ``improvement_flag``, and
   ``suppressing_post`` are not in ``positive_flags``. Review this choice when
   creating a new phenotype. Also decide whether improvement, negation, or
   suppression should veto an otherwise positive result.

6. Keep configuration and code aligned.
   If a tree introduces a new body region, synonym, keyword, or surgical
   progression chain, first add it to the appropriate section of
   ``pain_rules.yaml`` and test extraction. If it only changes how existing
   features are combined (for example, changing a threshold from 8 to 7), edit
   the decision-tree function only.

7. Validate before using the tree on a cohort.
   Create small, de-identified test cases for:

   - every rule firing by itself;
   - values immediately below, at, and above each threshold;
   - preoperative versus postoperative comparisons;
   - negated, historical, improving, and suppressing language;
   - regions inside and outside the relevant surgical chain;
   - left/right, bilateral, unknown, and missing laterality;
   - missing, malformed, duplicate, and out-of-range scores;
   - cases in which several branches fire together.

   Assert both ``predicted`` and the relevant entries in ``flags`` and
   ``trace``. Then evaluate the proposed tree against a manually reviewed,
   de-identified validation set and report the tree version, rules-file
   version, cohort, and performance. A regex phenotype is study logic, not a
   clinical diagnosis, and should be locally reviewed before reuse.

The trace is intentionally part of the public output: it lets users see which
branch fired and which extracted values were considered. Preserve or extend it
when customizing a tree.
"""

from __future__ import annotations

from typing import Dict, List, Any, Optional, Set, Union

from chronic_pain_nlp.extractor import (
    compute_region_progression_details,
    canonicalize_region_surface,
    SURGERY_TO_CHAINS,
)


def canon(r):
    if not isinstance(r, str):
        return None
    return r.strip().lower()


def _is_full_lr_flip(pre_set: set[str], post_set: set[str]) -> bool:
    return (pre_set == {"L"} and post_set == {"R"}) or (pre_set == {"R"} and post_set == {"L"})


def _normalize_region_of_surgery(region_of_surgery: str | set | None) -> set[str]:
    if isinstance(region_of_surgery, str):
        c = canon(region_of_surgery)
        return {c} if c else set()
    if region_of_surgery is None:
        return set()

    out: set[str] = set()
    for r in region_of_surgery:
        c = canon(r)
        if c:
            out.add(c)
    return out


def _canon_atomic_region_from_item(item: dict) -> Optional[str]:
    raw = item.get("atomic_region") or item.get("region")
    if not isinstance(raw, str):
        return None
    s = raw.strip().lower()
    if not s:
        return None
    return canonicalize_region_surface(s)


def _atomic_region_set_from_inputs(regional_scores: list | None, regions: list | None) -> set[str]:
    out: set[str] = set()

    for x in (regional_scores or []):
        if not isinstance(x, dict):
            continue
        r = _canon_atomic_region_from_item(x)
        if r:
            out.add(r)

    for r in (regions or []):
        if not isinstance(r, str):
            continue
        s = r.strip().lower()
        if not s:
            continue
        out.add(canonicalize_region_surface(s))

    return out


def _allowed_atomic_regions_for_surgery(region_of_surgery_set: set[str]) -> set[str]:
    """
    Build the allowed atomic regions directly from extractor.SURGERY_TO_CHAINS.
    region_of_surgery_set is normalized to lowercase here, so map back to upper keys.
    """
    allowed: set[str] = set()

    for surg in region_of_surgery_set:
        for chain in SURGERY_TO_CHAINS.get(str(surg).upper(), []):
            for r in chain:
                allowed.add(canonicalize_region_surface(r))

    return allowed


def _region_in_surgical_chain(region_canon: str, region_of_surgery_set: set[str]) -> bool:
    if not region_canon or not region_of_surgery_set:
        return False
    allowed = _allowed_atomic_regions_for_surgery(region_of_surgery_set)
    return region_canon in allowed


def predict_chronic_pain_decision_tree(
    *,
    post_general_scores: list,
    post_regional_scores: list,
    post_regions: list,
    pre_regions: list,
    region_of_surgery: str | set | None,
    neuropathic_pre: bool = False,
    neuropathic_post: bool = False,
    pre_general_scores: list | None = None,
    pre_regional_scores: list | None = None,
    suppressing_post: bool = False,
    general_pain_metric: str = "max",
    improvement_post: bool = False,
    preop_laterality_map: Dict[str, List[str]] | None = None,
    postop_laterality_map: Dict[str, List[str]] | None = None,
    return_early: bool = False,
    include_trace: bool = True,
    neuropathic_region_progression_pre: bool = False,
    neuropathic_region_progression_post: bool = False,
    neuropathic_region_progression_details_post: dict | None = None,
) -> dict:
    """
    Decision tree updated to match extractor.py.

    Key behavior:
    - general_flag fires if post-op general pain >= 8
    - regional_increase_flag fires if:
        * post-op regional pain >= 8 AND region is in surgical chain
        OR
        * same in-chain region exists pre-op and post-op > pre-op
    - region_progression still uses extractor.compute_region_progression_details
    - laterality logic preserved
    """

    pre_general_scores = pre_general_scores or []
    pre_regional_scores = pre_regional_scores or []

    region_of_surgery_set = _normalize_region_of_surgery(region_of_surgery)

    flags: dict[str, Any] = {
        "general_flag": False,
        "regional_increase_flag": False,
        "new_non_surgical_region_flag": False,
        "neuropathic_flag": False,
        "region_progression_flag": False,
        "laterality_switch_flag": False,
        "neuropathic_region_progression_flag": False,
        "neuropathic_region_progression_details": None,
        "improvement_flag": False,
        "suppressing_post": bool(suppressing_post),
    }

    trace: list[dict[str, Any]] = []

    def _maybe_return(rule_name: str):
        if return_early:
            out = {"predicted": 1, "flags": flags}
            if include_trace:
                out["trace"] = trace
            return out
        return None

    def summarize(scores, metric):
        values = []
        for x in (scores or []):
            if isinstance(x, dict) and "score" in x:
                try:
                    s = float(x["score"])
                except Exception:
                    continue
                if 0 <= s <= 10:
                    values.append(s)

        if not values:
            return 0.0

        if metric == "mean":
            return float(sum(values) / len(values))
        return float(max(values))

    # -------------------------------------------------------------------------
    # 1) General CPSP threshold
    # -------------------------------------------------------------------------
    pre_general = summarize(pre_general_scores, general_pain_metric)
    post_general = summarize(post_general_scores, general_pain_metric)

    fired = bool(post_general >= 8)

    if include_trace:
        trace.append(
            {
                "rule": "general_cpsp_threshold",
                "pre_general": pre_general,
                "post_general": post_general,
                "metric": general_pain_metric,
                "fired": fired,
                "note": "Fires if post-op general pain >= 8 regardless of pre-op.",
            }
        )

    if fired:
        flags["general_flag"] = True
        early = _maybe_return("general_cpsp_threshold")
        if early is not None:
            return early

    # -------------------------------------------------------------------------
    # Prepare pre-op regional lookup, restricted to surgical chain
    # -------------------------------------------------------------------------
    pre_region_max: dict[str, float] = {}
    for item in (pre_regional_scores or []):
        if not isinstance(item, dict):
            continue

        r = _canon_atomic_region_from_item(item)
        if not r:
            continue

        if not _region_in_surgical_chain(r, region_of_surgery_set):
            continue

        try:
            s = float(item.get("score"))
        except Exception:
            continue

        if 0 <= s <= 10:
            pre_region_max[r] = max(pre_region_max.get(r, 0.0), s)

    pre_scored_regions = set(pre_region_max.keys())

    # -------------------------------------------------------------------------
    # 2a) Regional CPSP threshold / increase, restricted to surgical chain
    # -------------------------------------------------------------------------
    for item in (post_regional_scores or []):
        if not isinstance(item, dict):
            continue

        r = _canon_atomic_region_from_item(item)
        if not r:
            continue

        in_chain = _region_in_surgical_chain(r, region_of_surgery_set)

        try:
            post_s = float(item.get("score"))
        except Exception:
            continue

        if not (0 <= post_s <= 10):
            continue

        if not in_chain:
            if include_trace:
                trace.append(
                    {
                        "rule": "regional_increase",
                        "region": r,
                        "has_pre_score": False,
                        "pre_max": None,
                        "post_score": post_s,
                        "in_chain": False,
                        "fired": False,
                        "note": "Skipped because region is not in surgical chain.",
                    }
                )
            continue

        has_pre = r in pre_scored_regions
        pre_s = float(pre_region_max.get(r, 0.0))

        fired = bool((post_s >= 8) or (has_pre and post_s > pre_s))

        if include_trace:
            trace.append(
                {
                    "rule": "regional_increase",
                    "region": r,
                    "has_pre_score": has_pre,
                    "pre_max": pre_s,
                    "post_score": post_s,
                    "in_chain": True,
                    "fired": fired,
                    "note": "Regional rule restricted to surgical chain; fires for post-op score >= 8 or increase over pre-op in same region.",
                }
            )

        if fired:
            flags["regional_increase_flag"] = True
            early = _maybe_return("regional_increase")
            if early is not None:
                return early

    # -------------------------------------------------------------------------
    # 2b) New non-surgical regional pain
    # -------------------------------------------------------------------------
    post_scored_regions: set[str] = set()
    for item in (post_regional_scores or []):
        if not isinstance(item, dict):
            continue
        r = _canon_atomic_region_from_item(item)
        if r:
            post_scored_regions.add(r)

    for r in sorted(post_scored_regions):
        is_new = r not in pre_scored_regions
        in_chain = _region_in_surgical_chain(r, region_of_surgery_set)
        fired = bool(is_new and not in_chain)

        if include_trace:
            trace.append(
                {
                    "rule": "new_non_surgical_region",
                    "region": r,
                    "is_new": is_new,
                    "in_chain": in_chain,
                    "region_of_surgery_set": sorted(region_of_surgery_set),
                    "fired": fired,
                }
            )

        if fired:
            flags["new_non_surgical_region_flag"] = True
            early = _maybe_return("new_non_surgical_region")
            if early is not None:
                return early

    # -------------------------------------------------------------------------
    # 3) Neuropathic region progression
    # -------------------------------------------------------------------------
    fired = bool(neuropathic_region_progression_post and not neuropathic_region_progression_pre)

    if include_trace:
        trace.append(
            {
                "rule": "neuropathic_region_progression_post_only",
                "neuropathic_region_progression_pre": bool(neuropathic_region_progression_pre),
                "neuropathic_region_progression_post": bool(neuropathic_region_progression_post),
                "details_post": neuropathic_region_progression_details_post,
                "fired": fired,
                "note": "Uses extractor's surgery-anchored neuropathic progression signal.",
            }
        )

    if fired:
        flags["neuropathic_region_progression_flag"] = True
        flags["neuropathic_region_progression_details"] = neuropathic_region_progression_details_post
        early = _maybe_return("neuropathic_region_progression_post_only")
        if early is not None:
            return early

    # -------------------------------------------------------------------------
    # 4) Region progression
    # -------------------------------------------------------------------------
    pre_atomic_regions = _atomic_region_set_from_inputs(pre_regional_scores, pre_regions)
    post_atomic_regions = _atomic_region_set_from_inputs(post_regional_scores, post_regions)

    prog_flag, prog_details = compute_region_progression_details(
        pre_atomic_regions,
        post_atomic_regions,
        region_of_surgery_set,
    )

    fired = False
    if prog_flag and prog_details:
        post_region = canon(prog_details.get("post_region"))
        post_region = canonicalize_region_surface(post_region) if post_region else None
        fired = bool(post_region and post_region not in pre_atomic_regions)

    if include_trace:
        trace.append(
            {
                "rule": "region_progression",
                "pre_atomic_regions_n": len(pre_atomic_regions),
                "post_atomic_regions_n": len(post_atomic_regions),
                "region_of_surgery_set": sorted(region_of_surgery_set),
                "prog_flag": bool(prog_flag),
                "prog_details": prog_details,
                "fired": fired,
            }
        )

    if fired:
        flags["region_progression_flag"] = True
        flags["region_progression_details"] = prog_details
        early = _maybe_return("region_progression")
        if early is not None:
            return early

    # -------------------------------------------------------------------------
    # 5) Laterality switch
    # -------------------------------------------------------------------------
    def _to_set_map(d):
        out = {}
        for reg, lats in (d or {}).items():
            if not isinstance(lats, (list, set, tuple)):
                continue

            vv = {str(x).strip().upper() for x in lats if str(x).strip()}
            vv.discard("U")

            if "B" in vv:
                vv |= {"L", "R"}
                vv.discard("B")

            out[str(reg).strip().lower()] = vv
        return out

    def _is_explicit_lr_flip(pre_lats: set[str], post_lats: set[str]) -> bool:
        return (pre_lats == {"L"} and post_lats == {"R"}) or (pre_lats == {"R"} and post_lats == {"L"})

    pre_lat = _to_set_map(preop_laterality_map)
    post_lat = _to_set_map(postop_laterality_map)

    switched = []
    overlap = sorted(set(pre_lat.keys()) & set(post_lat.keys()))
    for reg in overlap:
        pre_s = pre_lat.get(reg, set())
        post_s = post_lat.get(reg, set())

        fired_reg = _is_explicit_lr_flip(pre_s, post_s)

        if include_trace:
            trace.append(
                {
                    "rule": "laterality_explicit_lr_flip",
                    "region": reg,
                    "pre": sorted(pre_s),
                    "post": sorted(post_s),
                    "fired": fired_reg,
                    "note": "Counts only {L}->{R} or {R}->{L}.",
                }
            )

        if fired_reg:
            switched.append({"region": reg, "pre": sorted(pre_s), "post": sorted(post_s)})

    fired = bool(switched)

    if fired:
        flags["laterality_switch_flag"] = True
        flags["laterality_switch_details"] = {"switched": switched}
        early = _maybe_return("laterality_explicit_lr_flip")
        if early is not None:
            return early

    if improvement_post:
        flags["improvement_flag"] = True
        if include_trace:
            trace.append({"rule": "improvement_post", "improvement_post": True, "fired": False})

    positive_flags = [
        "general_flag",
        "regional_increase_flag",
        "neuropathic_region_progression_flag",
        "region_progression_flag",
        "laterality_switch_flag",
    ]
    predicted = int(any(bool(flags.get(k)) for k in positive_flags))

    out = {"predicted": predicted, "flags": flags}
    if include_trace:
        out["trace"] = trace
    return out


def predict_chronic_pain_region_of_surgery(
    *,
    pre_regional_scores: list | None,
    post_regional_scores: list | None,
    pre_regions: list | None,
    post_regions: list | None,
    region_of_surgery: str | set | None,
    preop_laterality_map: Dict[str, List[str]] | None = None,
    postop_laterality_map: Dict[str, List[str]] | None = None,
    neuropathic_region_progression_flag: bool = False,
    neuropathic_region_progression_details: dict | None = None,
    include_trace: bool = True,
) -> dict:
    pre_atomic_regions = _atomic_region_set_from_inputs(pre_regional_scores, pre_regions)
    post_atomic_regions = _atomic_region_set_from_inputs(post_regional_scores, post_regions)

    prog_flag, prog_details = compute_region_progression_details(
        pre_atomic_regions,
        post_atomic_regions,
        region_of_surgery,
    )

    fired_prog = False
    if prog_flag and prog_details:
        post_region = (prog_details.get("post_region") or "").strip().lower()
        post_region = canonicalize_region_surface(post_region) if post_region else None
        fired_prog = bool(post_region and post_region not in pre_atomic_regions)

    def _to_set_map(d: Dict[str, List[str]] | None) -> dict[str, set[str]]:
        out: dict[str, set[str]] = {}
        for reg, lats in (d or {}).items():
            if not isinstance(lats, (list, set, tuple)):
                continue
            vv = {str(x).strip().upper() for x in lats if str(x).strip()}
            vv.discard("U")
            if "B" in vv:
                vv |= {"L", "R"}
                vv.discard("B")
            out[str(reg).strip().lower()] = vv
        return out

    pre_lat = _to_set_map(preop_laterality_map)
    post_lat = _to_set_map(postop_laterality_map)
    overlap_regs = set(pre_lat.keys()) & set(post_lat.keys())

    switched = []
    for reg in sorted(overlap_regs):
        if _is_full_lr_flip(pre_lat[reg], post_lat[reg]):
            switched.append({"region": reg, "pre": sorted(pre_lat[reg]), "post": sorted(post_lat[reg])})

    fired_lat = bool(switched)
    fired_neuro_prog = bool(neuropathic_region_progression_flag)

    flags: dict[str, Any] = {
        "region_progression_flag": bool(fired_prog),
        "region_progression_details": prog_details if fired_prog else None,
        "laterality_switch_flag": bool(fired_lat),
        "laterality_switch_details": {"switched": switched} if fired_lat else None,
        "neuropathic_region_progression_flag": bool(fired_neuro_prog),
        "neuropathic_region_progression_details": (
            neuropathic_region_progression_details if fired_neuro_prog else None
        ),
    }

    predicted = int(fired_prog or fired_lat or fired_neuro_prog)

    out = {"predicted": predicted, "flags": flags}

    if include_trace:
        out["trace"] = [
            {
                "rule": "region_progression_only",
                "pre_atomic_regions_n": len(pre_atomic_regions),
                "post_atomic_regions_n": len(post_atomic_regions),
                "region_of_surgery": region_of_surgery,
                "prog_flag_raw": bool(prog_flag),
                "prog_details_raw": prog_details,
                "fired": bool(fired_prog),
            },
            {
                "rule": "laterality_full_lr_flip",
                "overlap_regions_considered": sorted(overlap_regs),
                "switched": switched,
                "fired": bool(fired_lat),
            },
            {
                "rule": "neuropathic_region_progression_from_extractor",
                "flag_in": bool(neuropathic_region_progression_flag),
                "details_in": neuropathic_region_progression_details,
                "fired": bool(fired_neuro_prog),
            },
        ]

    return out



