"""Build an offline HTML file containing all code, models, and WASM runtime."""

import argparse
import base64
import gzip
from pathlib import Path

from assets import ROOT, SOURCES, inline_js, tts_metadata, verify_vendor


def block(name: str, data: bytes) -> str:
    packed = base64.b64encode(gzip.compress(data, compresslevel=9, mtime=0)).decode()
    return f'<script type="application/octet-stream" data-name="{name}">{packed}</script>\n'


def build(models: Path, vendor: Path, out: Path, expected: Path | None = None) -> None:
    """Write a self-contained page; expected adds an optional Python parity check."""
    verify_vendor(vendor)
    blocks = [
        block("ort-mjs", (vendor / "ort-wasm-simd-threaded.mjs").read_bytes()),
        block("ort-wasm", (vendor / "ort-wasm-simd-threaded.wasm").read_bytes()),
        *(block(name, (models / f"{name}.onnx").read_bytes())
          for name in ("predictor", "acoustic", "vocoder", "bert")),
        block("stress-json", (models / "stress.json").read_bytes()),
        block("stress-bin", (models / "stress.bin").read_bytes()),
        block("tts-meta", tts_metadata(models)),
        block("voices", (ROOT / "web/voices.json").read_bytes()),
    ]
    if expected:
        blocks.append(block("expected", expected.read_bytes()))
    code = "\n".join("(() => {\n" + (ROOT / "src" / name).read_text(encoding="utf-8") + "\n})();"
                     for name in SOURCES)
    code += "\n" + (ROOT / "web/app.js").read_text(encoding="utf-8")
    runtime = inline_js((vendor / "ort.wasm.min.js").read_text(encoding="utf-8"))
    page = (ROOT / "web/index.html").read_text(encoding="utf-8")
    page = page.replace("<!-- PAYLOADS -->", "".join(blocks))
    page = page.replace("<!-- SCRIPTS -->", f"<script>\n{runtime}\n</script>\n<script>\n{inline_js(code)}\n</script>")
    notices = "<!--\n" + (ROOT / "LICENSE").read_text(encoding="utf-8") + "\n-->\n"
    notices += "\n".join(
        f"<!--\n{name}\n{(ROOT / 'licenses' / name).read_text(encoding='utf-8')}\n-->"
        for name in ("silero-cis.txt", "silero-stress.txt", "onnxruntime.txt"))
    page = page.replace("<!-- LICENSES -->", notices)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8", newline="\n")
    print(f"{out}: {out.stat().st_size / 1e6:.1f} MB")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, default=ROOT / "models")
    parser.add_argument("--vendor", type=Path, default=ROOT / "vendor")
    parser.add_argument("--out", type=Path, default=ROOT / "dist/silero.html")
    parser.add_argument("--expected", type=Path)
    args = parser.parse_args()
    try:
        build(args.models, args.vendor, args.out, args.expected)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"Build failed: {error}\n")


if __name__ == "__main__":
    main()
