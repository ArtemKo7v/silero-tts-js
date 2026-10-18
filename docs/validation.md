# Initial validation

Verified on 2026-10-18. Browser checks used desktop Chrome on Windows; model
exports ran in an independent Linux CPU virtual environment using the reference
versions in [export.md](export.md).

| Check | Result |
| --- | --- |
| JavaScript number and real-weight accentor tests | 85 passed, none skipped |
| Python packaging checks | 4 passed: roundtrip payloads, web assets, integrity rejection, metadata mismatch |
| Python compilation | All scripts compiled |
| Standalone HTML opened as `file:` | All 8 sample sentences synthesized; one WASM thread |
| HTTP demo with COOP/COEP | All 8 sample sentences synthesized; cross-origin isolation enabled |
| Browser/Python stress parity | 425 of 425 sentences matched |
| WAV validation in both browser modes | RIFF/WAVE, 24 kHz, mono PCM16, 882,600 samples, non-silent |
| Independent source export | All four ONNX graphs and both stress assets exactly matched the initial SHA-256 digests |
| Reconstructed BERT vs original TorchScript | Maximum difference 0.0 |
| Compressed vs fp32 TTS, 3 phrases and 2 voices | Equal lengths; audio SNR 42.3-56.3 dB |
| Compressed BERT vs fp32 hidden states | Maximum difference 0.001173 |
| Pitch vs original TorchScript, 5 levels and 2 voices | Equal lengths; equal reported median F0; SNR 43.6-56.3 dB |

The browser smoke test checks synthesis and WAV structure, not perceived audio
quality. These checks cover the sample text and reference corpus rather than
every possible input or browser. Generated models, corpora, logs, browser
profiles, screenshots, and HTML are excluded from Git.
