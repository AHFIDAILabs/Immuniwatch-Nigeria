import pytest
from unittest.mock import patch
from src.intelligence.counter import (
    CounterResponse,
    _build_prompt,
    _enforce_short,
    _enforce_word_limit,
    generate_counter_response,
    SHORT_MAX_CHARS,
    MEDIUM_MAX_WORDS,
    LONG_MAX_WORDS,
    LANGUAGE_NAMES,
)


# ---------------------------------------------------------------------------
# Configuration constants — system design Section 6.5
# ---------------------------------------------------------------------------

def test_short_max_chars():
    assert SHORT_MAX_CHARS == 280


def test_medium_max_words():
    assert MEDIUM_MAX_WORDS == 200


def test_long_max_words():
    assert LONG_MAX_WORDS == 500


def test_all_five_languages_supported():
    for code in ["en", "pcm", "ha", "yo", "ig"]:
        assert code in LANGUAGE_NAMES


# ---------------------------------------------------------------------------
# _enforce_short
# ---------------------------------------------------------------------------

def test_enforce_short_passes_text_under_limit():
    text = "Vaccines are safe. WHO confirms no link to infertility."
    result = _enforce_short(text)
    assert result == text


def test_enforce_short_truncates_at_word_boundary():
    text = "word " * 100  # well over 280 chars
    result = _enforce_short(text)
    assert len(result) <= SHORT_MAX_CHARS


def test_enforce_short_adds_ellipsis_when_truncated():
    text = "x " * 200
    result = _enforce_short(text)
    assert result.endswith("...")


def test_enforce_short_exactly_280_chars_passes():
    text = "a" * 280
    result = _enforce_short(text)
    assert len(result) == 280


# ---------------------------------------------------------------------------
# _enforce_word_limit
# ---------------------------------------------------------------------------

def test_enforce_word_limit_passes_under_limit():
    text = "vaccine is safe"
    result = _enforce_word_limit(text, 200)
    assert result == text


def test_enforce_word_limit_truncates_over_limit():
    text = " ".join(["word"] * 250)
    result = _enforce_word_limit(text, 200)
    words = result.replace("...", "").split()
    assert len(words) <= 200


def test_enforce_word_limit_adds_ellipsis():
    text = " ".join(["word"] * 300)
    result = _enforce_word_limit(text, 200)
    assert result.endswith("...")


# ---------------------------------------------------------------------------
# _build_prompt
# ---------------------------------------------------------------------------

def test_build_prompt_contains_claim():
    prompt = _build_prompt(
        claim="Vaccine causes infertility",
        language="en",
        evidence_snippets=["WHO: No evidence linking vaccines to infertility"],
        format_type="short",
    )
    assert "Vaccine causes infertility" in prompt


def test_build_prompt_contains_evidence():
    evidence = "WHO confirms vaccines are safe and effective"
    prompt = _build_prompt(
        claim="Vaccines are dangerous",
        language="en",
        evidence_snippets=[evidence],
        format_type="medium",
    )
    assert evidence in prompt


def test_build_prompt_contains_language_name():
    prompt = _build_prompt(
        claim="Vaccine dey kill pikin",
        language="pcm",
        evidence_snippets=["WHO: vaccines are safe"],
        format_type="short",
    )
    assert "Nigerian Pidgin" in prompt


def test_build_prompt_short_mentions_280_chars():
    prompt = _build_prompt(
        claim="Vaccine causes harm",
        language="en",
        evidence_snippets=["WHO: vaccines are safe"],
        format_type="short",
    )
    assert "280" in prompt


def test_build_prompt_medium_mentions_200_words():
    prompt = _build_prompt(
        claim="Vaccine causes harm",
        language="en",
        evidence_snippets=["WHO: vaccines are safe"],
        format_type="medium",
    )
    assert "200" in prompt


def test_build_prompt_long_mentions_500_words():
    prompt = _build_prompt(
        claim="Vaccine causes harm",
        language="en",
        evidence_snippets=["WHO: vaccines are safe"],
        format_type="long",
    )
    assert "500" in prompt


def test_build_prompt_instructs_no_fabrication():
    prompt = _build_prompt(
        claim="Vaccine causes harm",
        language="en",
        evidence_snippets=["WHO: vaccines are safe"],
        format_type="short",
    )
    assert "ONLY the evidence provided" in prompt


def test_build_prompt_limits_evidence_to_3_snippets():
    """Prompt should not include more than 3 evidence chunks."""
    snippets = [f"Evidence {i}" for i in range(10)]
    prompt = _build_prompt(
        claim="Vaccine causes harm",
        language="en",
        evidence_snippets=snippets,
        format_type="medium",
    )
    assert "Evidence 3" not in prompt


# ---------------------------------------------------------------------------
# CounterResponse dataclass
# ---------------------------------------------------------------------------

def test_counter_response_to_dict():
    resp = CounterResponse(
        post_id="test-001",
        original_claim="Vaccine causes infertility",
        language="en",
        short="Vaccines are safe. WHO confirms no link to infertility.",
        medium="The claim that vaccines cause infertility is false.",
        long="Multiple studies and WHO confirm vaccines do not affect fertility.",
        sources=["https://www.who.int"],
        provider="groq",
    )
    d = resp.to_dict()
    assert d["post_id"] == "test-001"
    assert d["language"] == "en"
    assert "short" in d
    assert "medium" in d
    assert "long" in d
    assert "sources" in d
    assert "provider" in d


# ---------------------------------------------------------------------------
# generate_counter_response — mocked LLM
# ---------------------------------------------------------------------------

MOCK_RESPONSE = "Vaccines are safe and effective. WHO confirms no evidence of harm."


@patch("src.intelligence.counter._generate", return_value=MOCK_RESPONSE)
def test_generate_returns_counter_response(mock_gen):
    result = generate_counter_response(
        post_id="test-001",
        claim="Vaccine causes infertility",
        language="en",
        evidence_snippets=["WHO: No evidence linking vaccines to infertility"],
        source_urls=["https://www.who.int"],
    )
    assert result is not None
    assert isinstance(result, CounterResponse)


@patch("src.intelligence.counter._generate", return_value=MOCK_RESPONSE)
def test_generate_short_within_limit(mock_gen):
    result = generate_counter_response(
        post_id="test-001",
        claim="Vaccine causes infertility",
        language="en",
        evidence_snippets=["WHO: vaccines are safe"],
        source_urls=[],
    )
    assert len(result.short) <= SHORT_MAX_CHARS


@patch("src.intelligence.counter._generate", return_value=MOCK_RESPONSE)
def test_generate_medium_within_word_limit(mock_gen):
    result = generate_counter_response(
        post_id="test-001",
        claim="Vaccine causes infertility",
        language="en",
        evidence_snippets=["WHO: vaccines are safe"],
        source_urls=[],
    )
    assert len(result.medium.split()) <= MEDIUM_MAX_WORDS + 1


@patch("src.intelligence.counter._generate", return_value=MOCK_RESPONSE)
def test_generate_post_id_preserved(mock_gen):
    result = generate_counter_response(
        post_id="my-post-xyz",
        claim="Vaccine causes harm",
        language="en",
        evidence_snippets=["WHO: vaccines are safe"],
        source_urls=[],
    )
    assert result.post_id == "my-post-xyz"


@patch("src.intelligence.counter._generate", return_value=MOCK_RESPONSE)
def test_generate_language_preserved(mock_gen):
    result = generate_counter_response(
        post_id="test-001",
        claim="Rigakafi yana haddasa matsala",
        language="ha",
        evidence_snippets=["WHO: vaccines are safe"],
        source_urls=[],
    )
    assert result.language == "ha"


@patch("src.intelligence.counter._generate", return_value=MOCK_RESPONSE)
def test_generate_sources_capped_at_5(mock_gen):
    result = generate_counter_response(
        post_id="test-001",
        claim="Vaccine causes harm",
        language="en",
        evidence_snippets=["WHO: vaccines are safe"],
        source_urls=[f"https://source{i}.com" for i in range(10)],
    )
    assert len(result.sources) <= 5


def test_generate_returns_none_for_empty_claim():
    result = generate_counter_response(
        post_id="test-001",
        claim="",
        language="en",
        evidence_snippets=["WHO: vaccines are safe"],
        source_urls=[],
    )
    assert result is None


def test_generate_returns_none_for_no_evidence():
    result = generate_counter_response(
        post_id="test-001",
        claim="Vaccine causes infertility",
        language="en",
        evidence_snippets=[],
        source_urls=[],
    )
    assert result is None


def test_generate_defaults_language_to_en_when_none():
    with patch("src.intelligence.counter._generate", return_value=MOCK_RESPONSE):
        result = generate_counter_response(
            post_id="test-001",
            claim="Vaccine causes harm",
            language=None,
            evidence_snippets=["WHO: vaccines are safe"],
            source_urls=[],
        )
    assert result.language == "en"