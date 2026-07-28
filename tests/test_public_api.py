from chronic_pain_nlp import analyze_note, analyze_pre_post


def test_analyze_note_returns_stable_object():
    result = analyze_note(
        "Synthetic example: left leg pain is 8/10 with burning discomfort.",
        region_of_surgery="LUMBAR",
    )
    assert isinstance(result.regions, list)
    assert isinstance(result.general_scores, list)
    assert isinstance(result.regional_scores, list)
    assert isinstance(result.to_dict(), dict)


def test_pre_post_returns_binary_prediction():
    result = analyze_pre_post(
        "Synthetic example: low back pain is 3/10.",
        "Synthetic example: left leg pain is 8/10.",
        region_of_surgery="LUMBAR",
    )
    assert result.predicted_chronic_pain in {0, 1}
    assert isinstance(result.flags, dict)
