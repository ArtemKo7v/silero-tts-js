# Local browser models

Place these exported files here (all ignored by Git):

| File | Approximate size | Purpose |
| --- | ---: | --- |
| `predictor.onnx` | 3.4 MB | Token duration and pitch prediction, fp32 |
| `acoustic.onnx` | 23.3 MB | Acoustic decoder, fp32 |
| `vocoder.onnx` | 29.3 MB | Vocos/PQMF vocoder, fp16 stored weights with fp32 computation |
| `bert.onnx` | 32.1 MB | Homograph BERT, int8 embeddings and fp16 stored weights |
| `stress.json` | 6.0 MB | Dictionaries, tokenizer, and weight layout |
| `stress.bin` | 2.8 MB | Accentor and homograph classifier weights |
| `meta.json` | 2 KB | Symbols, token IDs, and speaker IDs |
| `pitch.json` | 1 KB | Per-speaker pitch coefficients |

Optional: `expected.json`, a Python stress reference corpus for parity checks.
The original raw fp32 BERT export belongs in a separate staging directory;
`export_tts.py` produces the compressed browser version here.

To regenerate, see [export instructions](../docs/export.md). To check the initial
asset set on systems with GNU coreutils: `cd models && sha256sum -c checksums.sha256`.
Other compatible exports may differ byte-for-byte across tool versions.
