from pprint import pprint

from chronic_pain_nlp import analyze_pre_post

result = analyze_pre_post(
    preop_text="Synthetic example: low back pain is 3/10.",
    postop_text="Synthetic example: burning left leg pain is 8/10.",
    region_of_surgery="LUMBAR",
    enable_trace=True,
)

pprint(result.to_dict())
