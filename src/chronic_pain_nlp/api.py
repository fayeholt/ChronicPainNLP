"""Public, typed API for chronic-pain NLP extraction and pre/post comparison."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional

from .decision_tree import predict_chronic_pain_decision_tree
from .extractor import extract_pain_info, extract_pain_info_with_trace


@dataclass
class NeuropathicRegionProgression:
    flag: bool = False
    details: Optional[dict[str, Any]] = None


@dataclass
class PainExtraction:
    regions: list[str] = field(default_factory=list)
    general_scores: list[dict[str, Any]] = field(default_factory=list)
    regional_scores: list[dict[str, Any]] = field(default_factory=list)
    neuropathic_flag: bool = False
    neuropathic_hits: list[str] = field(default_factory=list)
    suppress_flag: bool = False
    improvement_flag: bool = False
    lateralized_regions: list[dict[str, Any]] = field(default_factory=list)
    laterality_map: dict[str, list[str]] = field(default_factory=dict)
    neuropathic_region_progression: NeuropathicRegionProgression = field(
        default_factory=NeuropathicRegionProgression
    )
    trace: Optional[dict[str, Any]] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PrePostAnalysis:
    preop: PainExtraction
    postop: PainExtraction
    predicted_chronic_pain: int
    flags: dict[str, Any]
    decision_trace: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _coerce_extraction(raw: tuple[Any, ...], *, trace_enabled: bool) -> PainExtraction:
    if trace_enabled:
        (
            regions, general, regional, neuropathic, hits, suppress, improvement,
            lateralized, laterality_map, trace, neuro_prog, neuro_details,
        ) = raw
    else:
        (
            regions, general, regional, neuropathic, hits, suppress, improvement,
            lateralized, laterality_map, neuro_prog, neuro_details,
        ) = raw
        trace = None

    return PainExtraction(
        regions=list(regions or []),
        general_scores=list(general or []),
        regional_scores=list(regional or []),
        neuropathic_flag=bool(neuropathic),
        neuropathic_hits=list(hits or []),
        suppress_flag=bool(suppress),
        improvement_flag=bool(improvement),
        lateralized_regions=list(lateralized or []),
        laterality_map=dict(laterality_map or {}),
        neuropathic_region_progression=NeuropathicRegionProgression(
            flag=bool(neuro_prog), details=neuro_details
        ),
        trace=trace,
    )


def analyze_note(
    text: str,
    *,
    region_of_surgery: Optional[str] = None,
    preop_regions: Optional[list[str] | set[str]] = None,
    enable_trace: bool = False,
) -> PainExtraction:
    """Extract pain-related concepts from one note.

    The caller is responsible for ensuring that text may legally and ethically be
    processed. The function does not persist input text.
    """
    if enable_trace:
        raw = extract_pain_info_with_trace(text, region_of_surgery=region_of_surgery)
    else:
        raw = extract_pain_info(
            text,
            region_of_surgery=region_of_surgery,
            preop_regions=preop_regions,
        )
    return _coerce_extraction(raw, trace_enabled=enable_trace)


def analyze_pre_post(
    preop_text: str,
    postop_text: str,
    *,
    region_of_surgery: Optional[str] = None,
    general_pain_metric: str = "max",
    enable_trace: bool = False,
) -> PrePostAnalysis:
    """Extract pre/post notes and apply the chronic-pain decision rules."""
    if general_pain_metric not in {"max", "mean"}:
        raise ValueError("general_pain_metric must be 'max' or 'mean'")

    pre = analyze_note(
        preop_text,
        region_of_surgery=region_of_surgery,
        enable_trace=enable_trace,
    )
    post = analyze_note(
        postop_text,
        region_of_surgery=region_of_surgery,
        preop_regions=set(pre.regions),
        enable_trace=enable_trace,
    )

    decision = predict_chronic_pain_decision_tree(
        post_general_scores=post.general_scores,
        post_regional_scores=post.regional_scores,
        post_regions=post.regions,
        pre_regions=pre.regions,
        region_of_surgery=region_of_surgery,
        neuropathic_pre=pre.neuropathic_flag,
        neuropathic_post=post.neuropathic_flag,
        pre_general_scores=pre.general_scores,
        pre_regional_scores=pre.regional_scores,
        suppressing_post=post.suppress_flag,
        general_pain_metric=general_pain_metric,
        improvement_post=post.improvement_flag,
        preop_laterality_map=pre.laterality_map,
        postop_laterality_map=post.laterality_map,
        include_trace=enable_trace,
        neuropathic_region_progression_pre=pre.neuropathic_region_progression.flag,
        neuropathic_region_progression_post=post.neuropathic_region_progression.flag,
        neuropathic_region_progression_details_post=post.neuropathic_region_progression.details,
    )

    return PrePostAnalysis(
        preop=pre,
        postop=post,
        predicted_chronic_pain=int(decision["predicted"]),
        flags=dict(decision.get("flags") or {}),
        decision_trace=list(decision.get("trace") or []),
    )
