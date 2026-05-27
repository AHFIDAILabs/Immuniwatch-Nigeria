"""
ImmuniWatch Nigeria — ONNX Model Export
========================================
Loads the DAPT base + LoRA adapter from HuggingFace,
merges the adapter into the base, exports to ONNX,
verifies correctness, benchmarks latency, and saves
the thresholds alongside the ONNX model.
"""

import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv

# Load .env file before reading any environment variables
load_dotenv()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration — confirmed from HuggingFace model cards
# ---------------------------------------------------------------------------
DAPT_REPO  = "AHFIDAILabs/immuniwatch-dapt-afro-xlmr"
LORA_REPO  = "AHFIDAILabs/immuniwatch-lora-classifier"
HF_TOKEN   = os.environ.get("HF_TOKEN", "")

# Confirmed from fine-tuning notebook (max_seq_len=128)
MAX_SEQ_LEN = 128

# Confirmed from model card — order matches LABEL_TO_ID in training
LABELS     = ["factual", "misinformation", "irrelevant"]
NUM_LABELS = len(LABELS)

OUTPUT_DIR      = Path("models/onnx")
ONNX_PATH       = OUTPUT_DIR / "immuniwatch_classifier.onnx"
THRESHOLDS_PATH = OUTPUT_DIR / "thresholds.json"
CONFIG_PATH     = OUTPUT_DIR / "model_config.json"

LATENCY_TARGET_MS = 80


# ---------------------------------------------------------------------------
# Step 1 — Load base + LoRA adapter, then merge
# ---------------------------------------------------------------------------
def load_and_merge():
    """
    Load DAPT base model and apply LoRA adapter from HuggingFace.
    Merge adapter weights into base before ONNX export.
    ONNX does not understand PEFT wrappers — merging is required.
    Returns (merged_model, tokenizer).
    """
    from peft import PeftModel
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if not HF_TOKEN:
        log.error("HF_TOKEN is not set.")
        log.error("Add HF_TOKEN=hf_xxxx to your .env file and try again.")
        sys.exit(1)

    log.info("Loading DAPT base: %s", DAPT_REPO)
    base = AutoModelForSequenceClassification.from_pretrained(
        DAPT_REPO,
        num_labels=NUM_LABELS,
        token=HF_TOKEN,
        ignore_mismatched_sizes=True,
    )

    log.info("Applying LoRA adapter: %s", LORA_REPO)
    peft_model = PeftModel.from_pretrained(base, LORA_REPO, token=HF_TOKEN)

    log.info("Merging LoRA weights into base...")
    model = peft_model.merge_and_unload()
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(LORA_REPO, token=HF_TOKEN)

    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    log.info("Model ready | %.1fM parameters", params_m)
    return model, tokenizer


# ---------------------------------------------------------------------------
# Step 2 — Export to ONNX
# ---------------------------------------------------------------------------
def export_onnx(model, tokenizer) -> float:
    """Export merged model to ONNX. Returns file size in MB."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    log.info("Exporting to ONNX (opset 14)...")

    dummy = tokenizer(
        "vaccine safety Nigeria",
        return_tensors="pt",
        max_length=MAX_SEQ_LEN,
        padding="max_length",
        truncation=True,
    )

    with torch.no_grad():
        torch.onnx.export(
            model,
            args=(dummy["input_ids"], dummy["attention_mask"]),
            f=str(ONNX_PATH),
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids":      {0: "batch_size"},
                "attention_mask": {0: "batch_size"},
                "logits":         {0: "batch_size"},
            },
            opset_version=14,
            do_constant_folding=True,
            verbose=False,
        )

    size_mb = ONNX_PATH.stat().st_size / 1024 / 1024
    log.info("ONNX saved: %s | %.1f MB", ONNX_PATH, size_mb)
    return size_mb


# ---------------------------------------------------------------------------
# Step 3 — Verify ONNX matches PyTorch
# ---------------------------------------------------------------------------
def verify(model, tokenizer) -> None:
    """
    Compare PyTorch and ONNX predictions on test sentences.
    Raises RuntimeError if any prediction disagrees.
    """
    import onnxruntime as ort

    log.info("Verifying ONNX predictions match PyTorch...")

    sentences = [
        "The COVID-19 vaccine is safe and effective for children.",
        "Vaccines contain microchips planted by Bill Gates.",
        "NPHCDA announces new immunisation campaign in Lagos.",
    ]

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    sess = ort.InferenceSession(str(ONNX_PATH), providers=providers)

    mismatches = 0
    for text in sentences:
        # PyTorch prediction
        pt_inputs = tokenizer(
            text, return_tensors="pt",
            max_length=MAX_SEQ_LEN, padding="max_length", truncation=True,
        )
        with torch.no_grad():
            pt_label = int(model(**pt_inputs).logits.argmax(dim=-1).item())

        # ONNX prediction
        np_inputs = tokenizer(
            text, return_tensors="np",
            max_length=MAX_SEQ_LEN, padding="max_length", truncation=True,
        )
        onnx_logits = sess.run(
            ["logits"],
            {
                "input_ids":      np_inputs["input_ids"].astype(np.int64),
                "attention_mask": np_inputs["attention_mask"].astype(np.int64),
            },
        )[0]
        onnx_label = int(np.argmax(onnx_logits, axis=-1)[0])

        match = pt_label == onnx_label
        log.info(
            "  %s  PT=%-16s ONNX=%-16s | %s",
            "PASS" if match else "FAIL",
            LABELS[pt_label],
            LABELS[onnx_label],
            text[:55],
        )
        if not match:
            mismatches += 1

    if mismatches:
        raise RuntimeError(
            f"ONNX verification failed: {mismatches} mismatch(es). "
            "Do not use this export for inference."
        )
    log.info("Verification passed.")


# ---------------------------------------------------------------------------
# Step 4 — Benchmark latency
# ---------------------------------------------------------------------------
def benchmark(tokenizer) -> float:
    """
    Benchmark inference latency on CPU over 100 runs.
    Returns average latency in milliseconds.
    """
    import onnxruntime as ort

    log.info("Benchmarking latency (100 runs, CPU)...")

    sess = ort.InferenceSession(
        str(ONNX_PATH), providers=["CPUExecutionProvider"]
    )
    inputs = tokenizer(
        "Vaccine go kill your pikin na government trick.",
        return_tensors="np",
        max_length=MAX_SEQ_LEN,
        padding="max_length",
        truncation=True,
    )
    ort_inputs = {
        "input_ids":      inputs["input_ids"].astype(np.int64),
        "attention_mask": inputs["attention_mask"].astype(np.int64),
    }

    # Warmup
    for _ in range(5):
        sess.run(["logits"], ort_inputs)

    n = 100
    t0 = time.perf_counter()
    for _ in range(n):
        sess.run(["logits"], ort_inputs)
    avg_ms = (time.perf_counter() - t0) / n * 1000

    log.info(
        "Latency: %.1f ms | target < %d ms | %s",
        avg_ms,
        LATENCY_TARGET_MS,
        "PASS" if avg_ms < LATENCY_TARGET_MS else "SLOW — GPU recommended",
    )
    return avg_ms


# ---------------------------------------------------------------------------
# Step 5 — Save thresholds and model config
# ---------------------------------------------------------------------------
def save_artifacts(size_mb: float, avg_ms: float, tokenizer) -> None:
    """
    Download thresholds.json from HuggingFace LoRA repo and save it
    alongside the ONNX model. Write model_config.json for inference server.
    """
    from huggingface_hub import hf_hub_download

    log.info("Downloading thresholds.json from %s...", LORA_REPO)
    local_path = hf_hub_download(
        repo_id=LORA_REPO,
        filename="thresholds.json",
        token=HF_TOKEN,
    )
    with open(local_path) as f:
        thresholds = json.load(f)

    with open(THRESHOLDS_PATH, "w") as f:
        json.dump(thresholds, f, indent=2)
    log.info("Thresholds saved: %s", THRESHOLDS_PATH)

    config = {
        "dapt_repo":          DAPT_REPO,
        "lora_repo":          LORA_REPO,
        "onnx_path":          str(ONNX_PATH),
        "thresholds_path":    str(THRESHOLDS_PATH),
        "labels":             LABELS,
        "num_labels":         NUM_LABELS,
        "max_seq_len":        MAX_SEQ_LEN,
        "vocab_size":         tokenizer.vocab_size,
        "onnx_size_mb":       round(size_mb, 1),
        "avg_latency_ms":     round(avg_ms, 1),
        "latency_target_ms":  LATENCY_TARGET_MS,
    }
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)
    log.info("Config saved: %s", CONFIG_PATH)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    log.info("=" * 60)
    log.info("ImmuniWatch Nigeria — ONNX Export")
    log.info("=" * 60)

    model, tokenizer = load_and_merge()
    size_mb          = export_onnx(model, tokenizer)
    verify(model, tokenizer)
    avg_ms           = benchmark(tokenizer)
    save_artifacts(size_mb, avg_ms, tokenizer)

    log.info("=" * 60)
    log.info("Done")
    log.info("  ONNX:       %s", ONNX_PATH)
    log.info("  Thresholds: %s", THRESHOLDS_PATH)
    log.info("  Config:     %s", CONFIG_PATH)
    log.info("  Size:       %.1f MB", size_mb)
    log.info("  Latency:    %.1f ms", avg_ms)
    log.info("=" * 60)


if __name__ == "__main__":
    main()