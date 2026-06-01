
import logging
import os
from dataclasses import dataclass
from typing import List, Optional

from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROVIDER      = os.environ.get("COUNTER_RESPONSE_PROVIDER", "groq").lower()
GROQ_API_KEY  = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL    = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

# Format limits — Section 6.5
SHORT_MAX_CHARS = 280
MEDIUM_MAX_WORDS = 200
LONG_MAX_WORDS   = 500

# Language display names for prompts
LANGUAGE_NAMES = {
    "en":  "English",
    "pcm": "Nigerian Pidgin",
    "ha":  "Hausa",
    "yo":  "Yoruba",
    "ig":  "Igbo",
}


# ---------------------------------------------------------------------------
# Response dataclass
# ---------------------------------------------------------------------------
@dataclass
class CounterResponse:
    post_id:        str
    original_claim: str
    language:       str
    short:          str    # ≤ 280 characters — Twitter/WhatsApp
    medium:         str    # ≤ 200 words — Facebook post
    long:           str    # ≤ 500 words — detailed article
    sources:        List[str]
    provider:       str

    def to_dict(self) -> dict:
        return {
            "post_id":        self.post_id,
            "original_claim": self.original_claim,
            "language":       self.language,
            "short":          self.short,
            "medium":         self.medium,
            "long":           self.long,
            "sources":        self.sources,
            "provider":       self.provider,
        }


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------
def _build_prompt(
    claim: str,
    language: str,
    evidence_snippets: List[str],
    format_type: str,
) -> str:
    lang_name = LANGUAGE_NAMES.get(language, "English")

    if format_type == "short":
        length_instruction = f"Write in {lang_name}. Maximum 280 characters. Be direct and factual."
    elif format_type == "medium":
        length_instruction = f"Write in {lang_name}. Maximum 200 words. Include context and a source reference."
    else:
        length_instruction = f"Write in {lang_name}. Maximum 500 words. Be thorough with explanation and sources."

    if evidence_snippets:
        evidence_block = "VERIFIED EVIDENCE FROM WHO/NPHCDA:\n" + "\n".join(
            f"- {snippet[:300]}" for snippet in evidence_snippets[:3]
        ) + "\n\nTASK: Write a counter-response that:\n1. Directly addresses the claim with facts\n2. Uses ONLY the evidence provided above — do not add facts not in the evidence\n3. Is culturally appropriate for Nigerian audiences\n4. Does not use technical jargon\n5. " + length_instruction
    else:
        evidence_block = (
            "TASK: Write a counter-response that:\n"
            "1. Directly addresses the claim using established vaccine science\n"
            "2. References WHO guidelines and NPHCDA vaccination programmes where relevant\n"
            "3. Is specific to the Nigerian public health context\n"
            "4. Is culturally appropriate for Nigerian audiences\n"
            "5. Does not use technical jargon\n"
            f"6. {length_instruction}"
        )

    return f"""You are a public health communication expert for Nigeria's NPHCDA.
A vaccine misinformation claim needs a factual counter-response.

CLAIM: {claim}

{evidence_block}

Write only the counter-response text. No preamble, no labels."""


# ---------------------------------------------------------------------------
# LLM call — Groq
# ---------------------------------------------------------------------------
def _call_groq(prompt: str) -> str:
    from groq import Groq
    client = Groq(api_key=GROQ_API_KEY)
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1024,
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# LLM call — Anthropic
# ---------------------------------------------------------------------------
def _call_anthropic(prompt: str) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text.strip()


# ---------------------------------------------------------------------------
# LLM dispatcher
# ---------------------------------------------------------------------------
def _generate(prompt: str) -> str:
    if PROVIDER == "anthropic":
        if not ANTHROPIC_KEY:
            raise ValueError("ANTHROPIC_API_KEY not set in .env")
        return _call_anthropic(prompt)
    else:
        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY not set in .env")
        return _call_groq(prompt)


# ---------------------------------------------------------------------------
# Length enforcement
# ---------------------------------------------------------------------------
def _enforce_short(text: str) -> str:
    if len(text) <= SHORT_MAX_CHARS:
        return text
    # Reserve 3 chars for "..." so final result stays within 280
    truncated = text[:SHORT_MAX_CHARS - 3]
    last_space = truncated.rfind(" ")
    return truncated[:last_space].rstrip() + "..." if last_space > 0 else truncated + "..."


def _enforce_word_limit(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + "..."


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def generate_counter_response(
    post_id:           str,
    claim:             str,
    language:          Optional[str],
    evidence_snippets: List[str],
    source_urls:       List[str],
) -> Optional[CounterResponse]:
    if not claim or len(claim.strip()) < 5:
        log.warning("Claim too short to generate counter-response")
        return None

    lang = language or "en"

    try:
        short_response = _enforce_short(
            _generate(_build_prompt(claim, lang, evidence_snippets, "short"))
        )
        medium_response = _enforce_word_limit(
            _generate(_build_prompt(claim, lang, evidence_snippets, "medium")),
            MEDIUM_MAX_WORDS,
        )
        long_response = _enforce_word_limit(
            _generate(_build_prompt(claim, lang, evidence_snippets, "long")),
            LONG_MAX_WORDS,
        )

        log.info(
            "Counter-response generated: post_id=%s lang=%s provider=%s",
            post_id, lang, PROVIDER,
        )

        return CounterResponse(
            post_id=        post_id,
            original_claim= claim,
            language=       lang,
            short=          short_response,
            medium=         medium_response,
            long=           long_response,
            sources=        source_urls[:5],
            provider=       PROVIDER,
        )

    except Exception as e:
        log.error("Counter-response generation failed for %s: %s", post_id, e)
        return None