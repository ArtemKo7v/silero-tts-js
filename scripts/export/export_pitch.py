"""Export pitch coefficients and compare ONNX pitch controls with original TorchScript.

Usage: python -I scripts/export/export_pitch.py MODEL_PT ONNX_DIR OUT_JSON
Tests five pitch multipliers for two voices and reports lengths, SNR, and median F0."""

import json
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

LEVELS = (0.75, 0.85, 1.0, 1.15, 1.25)
TEXT = "Здр+авствуйте, так звуч+ит +этот г+олос. Посл+ушайте, подх+одит ли он для в+ашей кн+иги."


def f0_median(audio: np.ndarray, rate: int = 24000) -> float:
    frame, hop, low, high = int(0.04 * rate), int(0.01 * rate), rate // 400, rate // 60
    values = []
    for start in range(0, len(audio) - frame, hop):
        chunk = audio[start:start + frame] - audio[start:start + frame].mean()
        energy = float(np.dot(chunk, chunk))
        if energy < 1e-3:
            continue
        corr = np.correlate(chunk, chunk, "full")[frame - 1:]
        lag = low + int(np.argmax(corr[low:high]))
        if corr[lag] / energy > 0.5:
            values.append(rate / lag)
    return round(float(np.median(values)), 1) if values else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description="Export pitch coefficients and compare ONNX with TorchScript.")
    parser.add_argument("model", type=Path)
    parser.add_argument("onnx_dir", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    model_pt, onnx_dir, out = str(args.model), args.onnx_dir, args.out
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import onnxruntime as ort
    from graphs import frame_tokens, postprocess_durations
    from export_tts import irfft_matrices

    torch.set_grad_enabled(False)
    package = torch.package.PackageImporter(model_pt).load_pickle("tts_models", "model").packages[0]
    jm = package.models[0]
    coefs = [float(v) for v in jm.mean_std_coef]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"mean_std_coef": coefs}), encoding="utf-8")
    speakers = {k: v for k, v in package.speaker_to_ids[0].items() if k.startswith("ru_")}
    sessions = {n: ort.InferenceSession(str(onnx_dir / f"{n}.onnx"), providers=["CPUExecutionProvider"])
                for n in ("predictor", "acoustic", "vocoder")}
    cos, sin = irfft_matrices()
    sequence = package.preprocess_tacotron(package.prepare_text_input(TEXT, "ru")[0])[0].unsqueeze(0)
    for voice in ("ru_aigul", "ru_dmitriy"):
        speaker = torch.tensor([speakers[voice]])
        for k in LEVELS:
            ones = torch.ones(sequence.shape)
            ref = jm(sequence, speaker, 24000, {}, ones, ones * k, None, None, "cpu")[0][0].numpy()
            log_dur, pitch = sessions["predictor"].run(None, {"sequence": sequence.numpy(), "speaker": speaker.numpy()})
            # As in src/tts.js, unvoiced pitch values below 0.001 have already been zeroed.
            shifted = np.where(pitch == 0, 0, pitch * k + (k - 1) * coefs[speakers[voice]]).astype(np.float32)
            (mel,) = sessions["acoustic"].run(None, {"sequence": sequence.numpy(), "speaker": speaker.numpy(),
                                                     "pitch": shifted, "frame_token": frame_tokens(postprocess_durations(log_dur))})
            audio = sessions["vocoder"].run(None, {"mel": mel, "cos_t": cos, "sin_t": sin})[0]
            n = min(len(ref), len(audio))
            snr = 10 * np.log10(np.sum(ref[:n] ** 2) / max(np.sum((ref[:n] - audio[:n]) ** 2), 1e-20))
            print(f"{voice} k={k}: len {len(ref)}/{len(audio)}, SNR {snr:.1f} dB, F0 jm {f0_median(ref)} Hz, "
                  f"ONNX {f0_median(audio)} Hz", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
