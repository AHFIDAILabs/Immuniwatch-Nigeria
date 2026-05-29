import json
import logging
import re
import time
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Label order must match training — confirmed from model card
# ---------------------------------------------------------------------------
LABELS = ["factual", "misinformation", "irrelevant"]
LABEL_TO_ID = {label: idx for idx, label in enumerate(LABELS)}

# ---------------------------------------------------------------------------
# Nigerian states for location extraction from post text
# ---------------------------------------------------------------------------
NIGERIAN_STATES = [
    "Abia", "Adamawa", "Akwa Ibom", "Anambra", "Bauchi", "Bayelsa",
    "Benue", "Borno", "Cross River", "Delta", "Ebonyi", "Edo", "Ekiti",
    "Enugu", "Gombe", "Imo", "Jigawa", "Kaduna", "Kano", "Katsina",
    "Kebbi", "Kogi", "Kwara", "Lagos", "Nasarawa", "Niger", "Ogun",
    "Ondo", "Osun", "Oyo", "Plateau", "Rivers", "Sokoto", "Taraba",
    "Yobe", "Zamfara", "FCT", "Abuja",
]
# Pre-compile pattern for performance
_STATE_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(s) for s in NIGERIAN_STATES) + r")\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Language code normalisation
# Connectors may use different codes — normalise to API contract values
# ---------------------------------------------------------------------------
LANG_NORMALISE = {
    "hau": "ha", "ibo": "ig", "yor": "yo",
    "en": "en", "pcm": "pcm", "pcm_lexicon": "pcm",
    "ha": "ha", "ig": "ig", "yo": "yo",
}

# ---------------------------------------------------------------------------
# Nigerian language keyword detectors
# Used when connector does not provide a reliable language code.
# langdetect cannot distinguish Nigerian languages, so we use lexical cues.
# ---------------------------------------------------------------------------
_HA_RE = re.compile(
    r'\b(gaskia|rigakafi|rigakafin|mallam|sannu|nagode|inshallah|yanzu|yau|jiya|'
    r'lafiya|haba|karya|kowa|komai|hanya|birni|ruwa|abinci|zakar|wallahi|toh|barka|'
    r'gida|mutane|sarki|masha|bazum|bazoum|kasuwa|talaka|gwamnati|kasa|duniya|'
    r'Allah ya|masha allah|ya allah|ya gafarta|ya jikan|ya taimaka)\b',
    re.IGNORECASE,
)
_YO_CHAR_RE = re.compile(r'[ọẹàáèéêâ]')      # tonal diacritics = strong Yoruba signal
_YO_WORD_RE = re.compile(
    r'\b(naa|awon|fun|si|ati|tabi|jẹ|ọlọrun|ẹjọwọ|nínú|rẹ|wọn)\b',
    re.IGNORECASE,
)
_IG_CHAR_RE = re.compile(r'[ụị]')              # Igbo-specific diacritics
_IG_WORD_RE = re.compile(
    r'\b(nke|ndi|bụ|dị|nwere|ndị|maka|ihe|oge|ụlọ|chi)\b',
    re.IGNORECASE,
)
_PCM_RE = re.compile(
    r'\b(dey|wetin|wahala|abeg|sabi|pikin|naija|una|vex|comot|oga|dem say|no be|e don)\b',
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Module-level singletons — loaded once at startup
# ---------------------------------------------------------------------------
_session = None      # onnxruntime.InferenceSession
_tokenizer = None    # transformers tokenizer
_thresholds = None   # dict of class biases
_config = None       # model_config.json


def load(onnx_path: str, thresholds_path: str,
         config_path: str, tokenizer_repo: str,
         hf_token: Optional[str] = None) -> None:
    global _session, _tokenizer, _thresholds, _config

    import onnxruntime as ort
    from transformers import AutoTokenizer

    log.info("Loading ONNX model: %s", onnx_path)
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    _session = ort.InferenceSession(onnx_path, providers=providers)

    log.info("Loading tokenizer: %s", tokenizer_repo)
    _tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_repo,
        token=hf_token or None,
    )

    with open(thresholds_path) as f:
        _thresholds = json.load(f)
    log.info("Thresholds loaded: %s", _thresholds["class_biases"])

    with open(config_path) as f:
        _config = json.load(f)

    log.info("Classifier ready.")


def is_loaded() -> bool:
    return all(x is not None for x in [_session, _tokenizer, _thresholds])


def classify(text: str, language: Optional[str] = None,
             location: Optional[str] = None) -> dict:
    if not is_loaded():
        raise RuntimeError("Classifier not loaded. Call load() at startup.")

    t0 = time.perf_counter()
    max_seq_len = _config.get("max_seq_len", 128)

    # Tokenize
    inputs = _tokenizer(
        text,
        return_tensors="np",
        max_length=max_seq_len,
        padding="max_length",
        truncation=True,
    )

    # Run ONNX inference
    logits = _session.run(
        ["logits"],
        {
            "input_ids":      inputs["input_ids"].astype(np.int64),
            "attention_mask": inputs["attention_mask"].astype(np.int64),
        },
    )[0][0]  # shape: (num_labels,)

    # Apply per-class logit biases from thresholds.json
    biases = _thresholds["class_biases"]
    for label, bias in biases.items():
        idx = LABEL_TO_ID.get(label)
        if idx is not None:
            logits[idx] += bias

    # Softmax probabilities
    exp_logits = np.exp(logits - logits.max())
    probs = exp_logits / exp_logits.sum()

    label_id = int(np.argmax(probs))
    confidence = float(probs[label_id])

    # Shannon entropy normalised to [0, 1] — Section 3.3.3
    entropy_raw = float(-np.sum(probs * np.log(probs + 1e-9)))
    entropy = round(entropy_raw / np.log(len(LABELS)), 4)

    # Alternatives — other labels sorted by confidence
    alternatives = [
        {"label": LABELS[i], "confidence": round(float(probs[i]), 4)}
        for i in np.argsort(probs)[::-1]
        if i != label_id
    ]

    processing_ms = int((time.perf_counter() - t0) * 1000)

    return {
        "label":         LABELS[label_id],
        "confidence":    round(confidence, 4),
        "entropy":       entropy,
        "alternatives":  alternatives,
        "language":      _resolve_language(text, language),
        "state":         _resolve_state(text, location),
        "processing_ms": processing_ms,
    }


# ---------------------------------------------------------------------------
# Language resolution
# ---------------------------------------------------------------------------
def _resolve_language(text: str, provided: Optional[str]) -> Optional[str]:
    if provided:
        normed = LANG_NORMALISE.get(provided.lower(), provided.lower())
        # If connector explicitly signals a Nigerian language, trust it
        if normed in ("ha", "yo", "ig", "pcm"):
            return normed

    # Score each Nigerian language by keyword/character presence
    ha_score  = len(_HA_RE.findall(text))
    yo_score  = len(_YO_CHAR_RE.findall(text)) * 2 + len(_YO_WORD_RE.findall(text))
    ig_score  = len(_IG_CHAR_RE.findall(text)) * 2 + len(_IG_WORD_RE.findall(text))
    pcm_score = len(_PCM_RE.findall(text))

    best_lang  = max(("ha", ha_score), ("yo", yo_score),
                     ("ig", ig_score), ("pcm", pcm_score),
                     key=lambda x: x[1])
    if best_lang[1] >= 2:
        return best_lang[0]

    # Fall back to langdetect for English detection
    try:
        from langdetect import detect
        detected = detect(text)
        return "en" if detected == "en" else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# State extraction
# ---------------------------------------------------------------------------
def _resolve_state(text: str, provided: Optional[str]) -> Optional[str]:
    if provided:
        match = _STATE_PATTERN.search(provided)
        if match:
            state = match.group(1).title()
            return "FCT" if state.lower() == "abuja" else state

    match = _STATE_PATTERN.search(text)
    if match:
        state = match.group(1).title()
        return "FCT" if state.lower() == "abuja" else state

    return None