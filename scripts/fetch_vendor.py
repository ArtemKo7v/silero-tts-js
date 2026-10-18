"""Download the pinned ONNX Runtime Web build and verify every SHA-256 digest."""

import argparse
import hashlib
from pathlib import Path
from urllib.request import urlopen

from assets import ROOT, VENDOR_SHA256, VENDOR_VERSION


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "vendor")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for name, digest in VENDOR_SHA256.items():
        target = args.out / name
        if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() == digest:
            print(f"Verified {name}")
            continue
        url = f"https://cdn.jsdelivr.net/npm/onnxruntime-web@{VENDOR_VERSION}/dist/{name}"
        with urlopen(url, timeout=120) as response:
            data = response.read()
        if hashlib.sha256(data).hexdigest() != digest:
            raise SystemExit(f"SHA-256 mismatch: {name}; refusing to save")
        target.write_bytes(data)
        print(f"Downloaded {name}")


if __name__ == "__main__":
    main()
