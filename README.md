# Silero TTS JS

This code turns text into speech directly in your users' browsers.
Audio is generated on the client's device, so you don't need a server to generate
speech, and the text stays on the device.

**Try the demo:** 
 - [Russian, v5_cis_base](https://research.potochka.com/silero/).
 - [English, v3_en](https://research.potochka.com/silero/en.html).

The code in this repository uses the `v5_cis_base` model, but you can use any of the 
publicly available [Silero models](https://github.com/snakers4/silero-models). 

Just point your AI coding agent at this repository and ask it to add text to speech
to your project. It can read the code, examples, and instructions below to work out
how to use it.

Built with **Silero TTS v5_cis_base**,
**Silero Stress 1.5**, and **ONNX Runtime Web 1.20.1** (WASM). An independent
JavaScript port, with 29 voices, contextual number expansion, automatic stress
and homographs, pitch control, and 24 kHz mono WAV output. Text stays on the device.

Two builds share the same application code:

- **Web demo:** loads scripts, models, and WASM as separate files over HTTP.
- **Standalone HTML:** embeds everything in one file (~105 MB), opens directly
  from disk, and works offline. Without cross-origin isolation it uses one WASM thread.

## Quick start

Requires Python 3.10+ for packaging and serving; Node.js 18+ for unit tests.
Packaging uses only Python's standard library. A modern desktop browser with
WebAssembly, BigInt, and `DecompressionStream` is required; Chrome is used for
the integration checks. Model export additionally requires PyTorch and ONNX.

1. Prepare the eight files listed in [models/README.md](models/README.md), either
   by copying an existing compatible asset set or following the
   [export instructions](docs/export.md). Model weights are not stored in Git.
2. Download the pinned runtime and build:

```sh
python scripts/fetch_vendor.py
python scripts/build_web.py
python scripts/serve.py
```

Open [localhost:8000](http://127.0.0.1:8000), wait for the models to load, enter
Russian text, select a voice, and choose **Speak**. Use **Download WAV** to save it.
The server binds to localhost and enables cross-origin isolation for WASM threads.
The complete built directory can also be served by another static HTTP server;
without COOP/COEP headers the app falls back to one thread.

For an offline file:

```sh
python scripts/build_html.py
```

Open `dist/silero.html` directly. Both builders accept `--models`, `--vendor`,
`--out`, and optional `--expected models/expected.json` for a Python stress parity check.
Run any script with `--help` for its arguments. Build commands work from any
working directory; explicitly supplied relative paths refer to the current directory.

## Repository layout

| Path | Contents |
| --- | --- |
| `src/` | Browser libraries: number rules, text preparation, stress, and TTS inference |
| `web/` | Shared demo template, application loader, and voice labels |
| `scripts/export/` | Standalone source-model exporters and eager network reconstruction |
| `scripts/build_web.py` | Build a demo with separately loaded assets |
| `scripts/build_html.py` | Build a self-contained offline HTML file |
| `scripts/fetch_vendor.py`, `scripts/serve.py` | Verified runtime download and localhost server |
| `models/`, `vendor/` | Local generated/downloaded assets, ignored by Git |
| `dist/` | Build results, ignored by Git |
| `tests/` | Number/stress regressions, builder checks, and a real-browser smoke test |
| `licenses/` | Original third-party MIT notices |

Large weights and generated HTML belong in release assets, outside Git history.
`models/checksums.sha256` identifies the initial compatible model set; it is a
reference rather than a requirement for exports made with different tool versions.
See [docs/export.md](docs/export.md) for source URLs, the reference environment,
graph formats, and rebuild commands.

## Checks

```sh
node --test tests/*.test.cjs
python -m unittest discover -s tests -p "test_*.py"
python -m compileall -q scripts
```

The real-weight accentor test is skipped if its model files are absent. For
end-to-end verification with Chrome installed:

```sh
python scripts/build_web.py --expected models/expected.json
python scripts/serve.py
# In a second terminal:
node tests/smoke-browser.mjs http://127.0.0.1:8000/
node tests/smoke-browser.mjs dist/silero.html
```

Set `CHROME_PATH` if Chrome is not in a standard location. The smoke test uses
an isolated browser profile, checks the optional parity corpus, synthesizes the
sample text, and validates the WAV header and non-silent PCM output.
See [initial validation results](docs/validation.md), including independent source exports.

## Scope and license

This is an experimental Russian browser implementation of the CIS base model,
not an official Silero package. It exposes the 29 `ru_*` voices, uses heuristic
Russian number rules, and does not implement the full upstream SSML API or
other model families. Stop takes effect between sentences. Large models and
intermediate tensors require substantial browser memory; synthesis speed depends
on the CPU and available WASM threads.

[MIT](LICENSE), including the selected CIS base and Stress models under their
respective original notices. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
