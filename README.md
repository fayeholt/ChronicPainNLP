# ChronicPainNLP

A rule-based Python package for extracting pain regions, pain scores, neuropathic language, laterality, and pre/post progression signals from clinical text.

> **Research software only.** This package is not a medical device, does not provide medical advice, and must not be used as the sole basis for diagnosis, treatment, or patient care decisions.

## Privacy

This public repository contains **no clinical records, patient identifiers, MRNs, labels, or validation datasets**. Examples and tests use synthetic text only. The library processes strings supplied by the caller and does not intentionally persist note text.

## Installation

```bash
pip install chronic-pain-nlp
```

For a local checkout:

```bash
python -m pip install -e ".[dev]"
```

## Analyze one note

```python
from chronic_pain_nlp import analyze_note

result = analyze_note(
    "Synthetic example: left leg pain rated 8/10 with burning discomfort.",
    region_of_surgery="LUMBAR",
)

print(result.to_dict())
```

## Compare synthetic preoperative and postoperative notes

```python
from chronic_pain_nlp import analyze_pre_post

result = analyze_pre_post(
    preop_text="Synthetic example: mild low back pain rated 3/10.",
    postop_text="Synthetic example: left leg pain rated 8/10 with burning discomfort.",
    region_of_surgery="LUMBAR",
)

print(result.predicted_chronic_pain)
print(result.flags)
```

## Streamlit demo

Install the optional app dependencies and launch the local demo:

```bash
python -m pip install -e ".[app]"
streamlit run streamlit_app.py
```

The demo highlights extracted regions, general and regional pain scores, neuropathic terms, and laterality cues directly in the submitted text.

The demo supports a single-note extraction mode and a pre/post comparison mode. It does not intentionally save submitted text, but public hosting services may retain access logs. Do not submit protected health information to a public deployment.

## What the package extracts

- General and region-associated pain scores
- Anatomical pain-region mentions
- Neuropathic terms and persistence cues
- Negation, suppression, and improvement language
- Laterality and left/right switching
- Surgery-anchored regional progression
- A transparent rule trace when `enable_trace=True`

## Configuration

Rules live in `src/chronic_pain_nlp/rules/pain_rules.yaml`. An alternate rules file can be supplied with the `PAIN_RULES_PATH` environment variable.

## Responsible use

The included rules were developed for a specific research context and may not generalize across institutions, specialties, note styles, populations, or documentation systems. Validate performance on an appropriately governed local dataset and report subgroup performance and failure modes.

## Development

```bash
pytest
ruff check .
python -m build
```

## Citation

If you use this software in research, please cite. If you use or modify the included decision rules, please also describe the rules-file version, decision-tree definition, and any institution-specific changes in your methods.

## License

MIT. See License in repo.
