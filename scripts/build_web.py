"""Build an HTTP demo that loads JavaScript, model files, and WASM separately."""

import argparse
import json
import shutil
from pathlib import Path

from assets import ROOT, MODEL_FILES, SOURCES, VENDOR_SHA256, tts_metadata, verify_vendor


def build(models: Path, vendor: Path, out: Path, expected: Path | None = None) -> None:
    """Copy runtime assets and write a relative manifest for the shared app loader."""
    verify_vendor(vendor)
    metadata = tts_metadata(models)
    # Validate inputs before writing a partial build.
    for name in MODEL_FILES:
        if not (models / name).is_file():
            raise FileNotFoundError(models / name)
    if expected and not expected.is_file():
        raise FileNotFoundError(expected)
    for directory in ("src", "models", "vendor", "licenses"):
        (out / directory).mkdir(parents=True, exist_ok=True)
    for name in SOURCES:
        # Each library has its own scope, as in the single-file build.
        code = (ROOT / "src" / name).read_text(encoding="utf-8")
        (out / "src" / name).write_text("(() => {\n" + code + "\n})();\n", encoding="utf-8")
    shutil.copyfile(ROOT / "web/app.js", out / "app.js")
    for name in MODEL_FILES:
        shutil.copyfile(models / name, out / "models" / name)
    for name in VENDOR_SHA256:
        shutil.copyfile(vendor / name, out / "vendor" / name)
    for item in (ROOT / "licenses").glob("*.txt"):
        shutil.copyfile(item, out / "licenses" / item.name)
    shutil.copyfile(ROOT / "LICENSE", out / "LICENSE")
    shutil.copyfile(ROOT / "THIRD_PARTY_NOTICES.md", out / "THIRD_PARTY_NOTICES.md")
    shutil.copyfile(ROOT / "web/voices.json", out / "voices.json")
    (out / "models/tts-meta.json").write_bytes(metadata)
    manifest = {name: f"models/{name}.onnx" for name in ("predictor", "acoustic", "vocoder", "bert")}
    manifest.update({"ort-mjs": "vendor/ort-wasm-simd-threaded.mjs",
                     "ort-wasm": "vendor/ort-wasm-simd-threaded.wasm",
                     "stress-json": "models/stress.json", "stress-bin": "models/stress.bin",
                     "tts-meta": "models/tts-meta.json", "voices": "voices.json"})
    if expected:
        shutil.copyfile(expected, out / "models/expected.json")
        manifest["expected"] = "models/expected.json"
    else:
        (out / "models/expected.json").unlink(missing_ok=True)
    (out / "assets.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    scripts = ['<script src="vendor/ort.wasm.min.js"></script>']
    scripts.extend(f'<script src="src/{name}"></script>' for name in SOURCES)
    scripts.append('<script src="app.js"></script>')
    page = (ROOT / "web/index.html").read_text(encoding="utf-8")
    page = page.replace("<!-- PAYLOADS -->", "").replace("<!-- SCRIPTS -->", "\n".join(scripts))
    page = page.replace("<!-- LICENSES -->", '<!-- See THIRD_PARTY_NOTICES.md and licenses/. -->')
    (out / "index.html").write_text(page, encoding="utf-8", newline="\n")
    print(f"{out}: HTTP demo ready")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, default=ROOT / "models")
    parser.add_argument("--vendor", type=Path, default=ROOT / "vendor")
    parser.add_argument("--out", type=Path, default=ROOT / "dist/web")
    parser.add_argument("--expected", type=Path)
    args = parser.parse_args()
    try:
        build(args.models, args.vendor, args.out, args.expected)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"Build failed: {error}\n")


if __name__ == "__main__":
    main()
