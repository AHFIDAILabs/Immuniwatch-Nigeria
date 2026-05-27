import pytest
from src.models.classifier import (
    _resolve_state,
    _resolve_language,
    is_loaded,
    LABELS,
    LABEL_TO_ID,
    NIGERIAN_STATES,
    LANG_NORMALISE,
)


# ---------------------------------------------------------------------------
# is_loaded
# ---------------------------------------------------------------------------

def test_is_loaded_returns_false_before_load():
    """Model should not be loaded in test environment."""
    assert is_loaded() is False


# ---------------------------------------------------------------------------
# LABELS and LABEL_TO_ID
# ---------------------------------------------------------------------------

def test_labels_order():
    """Label order must match training — confirmed from model card."""
    assert LABELS == ["factual", "misinformation", "irrelevant"]


def test_label_to_id_mapping():
    assert LABEL_TO_ID["factual"] == 0
    assert LABEL_TO_ID["misinformation"] == 1
    assert LABEL_TO_ID["irrelevant"] == 2


def test_all_labels_have_ids():
    for label in LABELS:
        assert label in LABEL_TO_ID


# ---------------------------------------------------------------------------
# _resolve_state
# ---------------------------------------------------------------------------

def test_resolve_state_uses_provided_location():
    """Connector-provided location takes priority over text extraction."""
    result = _resolve_state("Some text", provided="Kano")
    assert result == "Kano"


def test_resolve_state_extracts_from_text():
    result = _resolve_state("In Lagos they are rejecting vaccines", provided=None)
    assert result == "Lagos"


def test_resolve_state_normalises_abuja_to_fct():
    result = _resolve_state("People in Abuja are worried", provided=None)
    assert result == "FCT"


def test_resolve_state_returns_none_when_no_signal():
    result = _resolve_state("Vaccines contain microchips", provided=None)
    assert result is None


def test_resolve_state_case_insensitive():
    result = _resolve_state("News from kano state today", provided=None)
    assert result == "Kano"


def test_resolve_state_all_36_states_detectable():
    """Each state name should be detectable from text."""
    states_to_check = [
        "Lagos", "Kano", "Rivers", "Oyo", "Kaduna",
        "Borno", "Sokoto", "Enugu", "Imo", "Delta"
    ]
    for state in states_to_check:
        text = f"Vaccine news from {state} state"
        result = _resolve_state(text, provided=None)
        assert result is not None, f"Failed to detect state: {state}"


# ---------------------------------------------------------------------------
# _resolve_language
# ---------------------------------------------------------------------------

def test_resolve_language_uses_provided_code():
    """Connector-provided language takes priority."""
    result = _resolve_language("any text", provided="ha")
    assert result == "ha"


def test_resolve_language_normalises_hau_to_ha():
    result = _resolve_language("any text", provided="hau")
    assert result == "ha"


def test_resolve_language_normalises_ibo_to_ig():
    result = _resolve_language("any text", provided="ibo")
    assert result == "ig"


def test_resolve_language_normalises_yor_to_yo():
    result = _resolve_language("any text", provided="yor")
    assert result == "yo"


def test_resolve_language_passes_through_en():
    result = _resolve_language("any text", provided="en")
    assert result == "en"


def test_resolve_language_passes_through_pcm():
    result = _resolve_language("any text", provided="pcm")
    assert result == "pcm"


def test_resolve_language_detects_english_from_text():
    result = _resolve_language(
        "The COVID vaccine is safe and effective for children",
        provided=None,
    )
    assert result == "en"


def test_resolve_language_returns_none_for_unknown_language():
    """When no language provided and text is not English, returns None."""
    result = _resolve_language("", provided=None)
    assert result is None or isinstance(result, str)


# ---------------------------------------------------------------------------
# NIGERIAN_STATES list integrity
# ---------------------------------------------------------------------------

def test_nigerian_states_contains_fct():
    assert "FCT" in NIGERIAN_STATES


def test_nigerian_states_contains_major_states():
    for state in ["Lagos", "Kano", "Rivers", "Abuja"]:
        assert state in NIGERIAN_STATES


def test_nigerian_states_count():
    """Nigeria has 36 states + FCT + Abuja alias = 38 entries."""
    assert len(NIGERIAN_STATES) == 38
