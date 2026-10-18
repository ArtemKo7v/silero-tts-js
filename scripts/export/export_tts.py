"""Export Silero v5_cis_base to browser ONNX graphs and compare compressed weights with fp32.

Usage: python -I scripts/export/export_tts.py MODEL_PT STRESS_DIR/bert.onnx OUT_DIR
The vocoder receives iRFFT matrices as inputs. Large vocoder and BERT initializers
use float16 storage with Cast to float32. Predictor and acoustic weights stay fp32.
Writes graphs, tokenizer metadata, pitch coefficients, and fp16 parity reports."""

import json
import argparse
import sys
from pathlib import Path

import numpy as np
import onnx
import torch
from onnx import helper, numpy_helper


def irfft_matrices(n_fft: int = 2400) -> tuple[np.ndarray, np.ndarray]:
    """Return float32 inverse real FFT matrices [n_fft, bins] for the browser vocoder."""
    bins = n_fft // 2 + 1
    n, k = np.arange(n_fft)[:, None], np.arange(bins)[None, :]
    angle = 2 * np.pi * k * n / n_fft
    weight = np.full(bins, 2.0)
    weight[0] = weight[-1] = 1.0
    cos = weight * np.cos(angle) / n_fft
    sin = weight * np.sin(angle) / n_fft
    sin[:, 0] = sin[:, -1] = 0
    return cos.astype(np.float32), sin.astype(np.float32)


def to_fp16(source: Path, target: Path, min_size: int = 4096) -> None:
    """Replace large float32 initializers with float16 storage and leading Cast nodes."""
    model = onnx.load(str(source))
    graph = model.graph
    casts, keep = [], []
    for init in graph.initializer:
        array = numpy_helper.to_array(init)
        if array.dtype == np.float32 and array.size >= min_size:
            half = numpy_helper.from_array(array.astype(np.float16), init.name + "__fp16")
            keep.append(half)
            casts.append(helper.make_node("Cast", [half.name], [init.name], to=onnx.TensorProto.FLOAT,
                                          name=init.name + "__cast"))
        else:
            keep.append(init)
    del graph.initializer[:]
    graph.initializer.extend(keep)
    nodes = list(graph.node)
    del graph.node[:]
    graph.node.extend(casts + nodes)
    onnx.checker.check_model(model)
    onnx.save(model, str(target))


def main() -> int:
    parser = argparse.ArgumentParser(description="Export browser TTS graphs and compare compressed weights with fp32.")
    parser.add_argument("model", type=Path)
    parser.add_argument("bert", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    model_pt, bert_onnx, out = str(args.model), args.bert.resolve(), args.out.resolve()
    if bert_onnx == out / "bert.onnx":
        parser.error("Use a separate stress export directory: the source BERT must remain fp32 for parity checks")
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    torch.set_grad_enabled(False)
    import onnxruntime as ort
    from graphs import PHRASES, Predictor, Vocoder, frame_tokens, postprocess_durations
    from silero_eager import load_from_jit

    class VocoderInputs(Vocoder):
        """Vocoder with inverse FFT matrices supplied as inputs instead of graph constants."""

        def forward(self, mel, cos_t, sin_t):
            x = self.out(self.backbone(mel)).transpose(1, 2)[0]
            mag, phase = torch.chunk(x, 2, 0)
            mag = torch.clamp(torch.exp(mag), max=100.0)
            frames = (cos_t @ (mag * torch.cos(phase)) - sin_t @ (mag * torch.sin(phase))) * self.window[:, None]
            y = self.overlap_add(frames)
            envelope = self.overlap_add((self.window ** 2)[:, None].expand(-1, frames.shape[1]))
            pad = (2400 - 600) // 2
            y = (y / envelope)[pad:-pad]
            bands = torch.nn.functional.conv1d(y[None, None], self.H, None, 2, 31)
            bands = torch.where(bands.abs().max() <= 1.0, bands, torch.tanh(bands))
            return bands[0, 0]

    package = torch.package.PackageImporter(model_pt).load_pickle("tts_models", "model").packages[0]
    jm = package.models[0]
    nets = load_from_jit(jm)
    predictor, acoustic = Predictor(nets).eval(), nets["tacotron"]
    vocoder = VocoderInputs(nets, jm).eval()
    del vocoder.cos_t, vocoder.sin_t  # Matrices arrive as inputs; these buffers are no longer needed.
    cos, sin = irfft_matrices()
    speakers = {k: v for k, v in package.speaker_to_ids[0].items() if k.startswith("ru_")}
    meta = {"symbols": package.symbols, "symbol_to_id": package.symbol_to_id, "speakers": speakers,
            "sos": package.sos_token, "eos": package.eos_token, "sample_rate": 24000}
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "pitch.json").write_text(json.dumps({"mean_std_coef": [float(v) for v in jm.mean_std_coef]}),
                                   encoding="utf-8")
    for name in ("stress.json", "stress.bin", "expected.json"):
        source = bert_onnx.parent / name
        if source.is_file():
            (out / name).write_bytes(source.read_bytes())

    def tokens(text):
        prepared, _, _ = package.prepare_text_input(text, "ru")
        sequence, _ = package.preprocess_tacotron(prepared)
        return sequence.unsqueeze(0)

    sequence, speaker = tokens(PHRASES[0]), torch.tensor([speakers["ru_aigul"]])
    log_dur, pitch = predictor(sequence, speaker)
    frame_token = torch.from_numpy(frame_tokens(postprocess_durations(log_dur.numpy())))
    mel = acoustic(sequence, speaker, pitch, frame_token)
    fp32 = out / "fp32"
    fp32.mkdir(exist_ok=True)
    exports = [
        ("predictor", predictor, (sequence, speaker), ["sequence", "speaker"], ["log_dur", "pitch"],
         {"sequence": {1: "T"}, "log_dur": {1: "T"}, "pitch": {2: "T"}}),
        ("acoustic", acoustic, (sequence, speaker, pitch, frame_token),
         ["sequence", "speaker", "pitch", "frame_token"], ["mel"],
         {"sequence": {1: "T"}, "pitch": {2: "T"}, "frame_token": {0: "F"}, "mel": {2: "F"}}),
        ("vocoder", vocoder, (mel, torch.from_numpy(cos), torch.from_numpy(sin)), ["mel", "cos_t", "sin_t"],
         ["audio"], {"mel": {2: "F"}, "audio": {0: "N"}}),
    ]
    for name, module, args, inputs, outputs, axes in exports:
        torch.onnx.export(module, args, str(fp32 / f"{name}.onnx"), input_names=inputs, output_names=outputs,
                          dynamic_axes=axes, opset_version=17, dynamo=False)
    for name in ("predictor", "acoustic"):
        (out / f"{name}.onnx").write_bytes((fp32 / f"{name}.onnx").read_bytes())
    to_fp16(fp32 / "vocoder.onnx", out / "vocoder.onnx")
    to_fp16(bert_onnx, out / "bert.onnx")

    def session(path):
        return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])

    def synth(directory, text, voice):
        s = {n: session(directory / f"{n}.onnx") for n in ("predictor", "acoustic", "vocoder")}
        seq, spk = tokens(text).numpy(), np.array([speakers[voice]], dtype=np.int64)
        log_dur, pitch = s["predictor"].run(None, {"sequence": seq, "speaker": spk})
        (mel,) = s["acoustic"].run(None, {"sequence": seq, "speaker": spk, "pitch": pitch,
                                          "frame_token": frame_tokens(postprocess_durations(log_dur))})
        return s["vocoder"].run(None, {"mel": mel, "cos_t": cos, "sin_t": sin})[0]

    report = {"sizes": {p.name: p.stat().st_size for p in sorted(out.glob("*.onnx"))}, "phrases": []}
    for text in PHRASES:
        for voice in ("ru_aigul", "ru_dmitriy"):
            ref, half = synth(fp32, text, voice), synth(out, text, voice)
            length = min(len(ref), len(half))
            noise = np.sum((ref[:length] - half[:length]) ** 2)
            report["phrases"].append({"voice": voice, "len_fp32": len(ref), "len_fp16": len(half),
                                      "snr_db": round(float(10 * np.log10(np.sum(ref[:length] ** 2) / max(noise, 1e-20))), 1)})
            np.save(out / f"fp16-{voice}-{PHRASES.index(text)}.npy", half)
    ids = np.random.default_rng(0).integers(5, 80000, (2, 60)).astype(np.int64)
    h32 = session(bert_onnx).run(None, {"input_ids": ids})[0]
    h16 = session(out / "bert.onnx").run(None, {"input_ids": ids})[0]
    report["bert_hidden_max_diff"] = float(np.abs(h32 - h16).max())
    (out / "fp16.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
