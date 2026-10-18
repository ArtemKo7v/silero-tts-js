# Exporting the source models

Packaging existing browser assets needs only Python's standard library. The
commands below are for regenerating those assets from the original models,
without any dependency on the original TTS Service repository.

## Environment and inputs

Use a separate Python virtual environment. The reference Linux CPU environment
uses `torch==2.14.1+cpu`, `numpy==2.5.4`, `onnx==1.23.2`,
`onnxruntime==1.31.0`, `onnxscript==0.7.2`, and `silero-stress==1.5`.
These are the original research environment's versions; other versions may
produce different serialized graphs and need their own parity checks.

```sh
python -m venv .venv
# Linux/macOS:
. .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r scripts/export/requirements.txt
```

The requirements install CPU PyTorch from the PyTorch wheel index. They describe
the reference environment, not a cross-platform lockfile.

Download the original `v5_cis_base.pt` from
[Silero's model host](https://models.silero.ai/models/tts/ru/v5_cis_base.pt)
to `inputs/v5_cis_base.pt`. The initial model SHA-256 is
`ba41b18f6a707ad93605a162998865e7c087153d2e010a26dd02229dab0e672a`.
Only use trusted source packages: loading a `torch.package` model executes its
packaged Python code. `silero-stress` downloads its own upstream data on first use.

## Generate browser assets

Run from the repository root with the export environment active:

```sh
# Export dictionaries, classifier weights, and fp32 homograph BERT.
python -I scripts/export/export_stress.py inputs/stress

# Export TTS, compress BERT/vocoder weights, write meta.json and pitch.json,
# and copy the stress dictionaries into models/.
python -I scripts/export/export_tts.py inputs/v5_cis_base.pt inputs/stress/bert.onnx models

# Optional: compare all five pitch levels with the original model for two voices.
python -I scripts/export/export_pitch.py inputs/v5_cis_base.pt models models/pitch.json

python scripts/fetch_vendor.py
python scripts/build_web.py
python scripts/build_html.py
```

`export_stress.py` compares the reconstructed BERT with the upstream TorchScript
model and requires maximum hidden-state difference below `1e-4`.
`export_tts.py` writes `fp16.json` and audio arrays with fp32/compressed comparison
results for three phrases and two voices; inspect the SNR and BERT differences.
`models/fp32/` contains intermediate graphs and is not needed by the browser.
The compact exports retain fp32 predictor/acoustic weights because compressing
the acoustic decoder noticeably changes the vocoder output.

## Optional reference corpus

Provide a UTF-8 file containing one sentence per line:

```sh
python -I scripts/export/export_stress.py inputs/stress --corpus inputs/corpus.txt
python -I scripts/export/export_tts.py inputs/v5_cis_base.pt inputs/stress/bert.onnx models
python scripts/build_html.py --out dist/silero-test.html --expected models/expected.json
python scripts/build_web.py --expected models/expected.json
```

For a larger corpus, `python scripts/export/make_corpus.py inputs/book.html inputs/corpus.txt 400`
extracts sentences from local cp1251 HTML and prepends difficult sample phrases.
The source text must be supplied separately with suitable reuse rights.
The reference checks the default stress mode before number normalization;
number normalization and the CIS monosyllabic mode have separate unit tests.

## Graph interface

| Graph | Inputs | Outputs |
| --- | --- | --- |
| Predictor | `sequence` int64 `[1,T]`, `speaker` int64 `[1]` | `log_dur` `[1,T]`, `pitch` `[1,1,T]` |
| Acoustic | `sequence`, `speaker`, `pitch`, `frame_token` int64 `[F]` | `mel` float32 `[1,192,F]` |
| Vocoder | `mel`, `cos_t`, `sin_t` float32 `[2400,1201]` | `audio` float32 `[N]`, 24 kHz |
| BERT | `input_ids` int64 `[B,L]` | `hidden` float32 `[B,L,312]` |

JavaScript handles text normalization, stress preservation, tokenization,
duration rounding, frame expansion, pitch adjustment, inverse FFT matrices,
paragraph pauses, and WAV assembly. ONNX graphs use opset 17.
