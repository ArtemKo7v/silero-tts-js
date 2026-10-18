"""Shared asset names, integrity checks, and metadata for both HTML builders."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENDOR_VERSION = "1.20.1"
VENDOR_SHA256 = {
    "ort.wasm.min.js": "e66568724f8848e57cc2a56e4bea3a5b86ce3ff81b2da11eac8ed7ab02b27bd4",
    "ort-wasm-simd-threaded.mjs": "745eb7c0ce6f18a6aa521971b2877babc7ffb27eecb58ab3bc6e5ef4692672e8",
    "ort-wasm-simd-threaded.wasm": "207d02be4591c156b0a98f024f3d58005b5b04c92274d759fb390338c63559ea",
}
MODEL_FILES = (
    "predictor.onnx", "acoustic.onnx", "vocoder.onnx", "bert.onnx",
    "stress.json", "stress.bin", "meta.json", "pitch.json",
)
SOURCES = ("numbers.js", "text.js", "stress.js", "tts.js")


def verify_vendor(directory: Path) -> None:
    """Reject missing or altered runtime files before distributing a build."""
    for name, expected in VENDOR_SHA256.items():
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"SHA-256 mismatch: {directory / name}")


def tts_metadata(models: Path) -> bytes:
    """Combine tokenizer, speaker IDs, and pitch coefficients into browser metadata."""
    meta = json.loads((models / "meta.json").read_text(encoding="utf-8"))
    result = {key: meta[key] for key in ("symbols", "symbol_to_id", "speakers", "sos", "eos")}
    result.update(json.loads((models / "pitch.json").read_text(encoding="utf-8")))
    voices = json.loads((ROOT / "web/voices.json").read_text(encoding="utf-8"))
    if {voice[0] for voice in voices} != set(result["speakers"]):
        raise ValueError("Voice labels do not match model speaker IDs")
    if len(result["mean_std_coef"]) <= max(result["speakers"].values()):
        raise ValueError("Missing pitch coefficients for model speakers")
    return json.dumps(result, ensure_ascii=False).encode()


def inline_js(code: str) -> str:
    """Escape HTML raw-text closing tags in embedded JavaScript."""
    return code.replace("</script", "<\\/script")
