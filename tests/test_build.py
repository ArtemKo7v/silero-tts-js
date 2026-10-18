"""Verify both packaging modes without requiring large model files."""

import base64
import gzip
import hashlib
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import assets
import build_html
import build_web


class BuildTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.models, self.vendor = self.root / "models", self.root / "vendor"
        self.models.mkdir()
        self.vendor.mkdir()
        for name in assets.MODEL_FILES:
            (self.models / name).write_bytes(f"fixture:{name}".encode())
        voices = json.loads((ROOT / "web/voices.json").read_text(encoding="utf-8"))
        meta = dict(symbols="abc", symbol_to_id={"a": 0}, sos="a", eos="b",
                    speakers={v[0]: i for i, v in enumerate(voices)})
        (self.models / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
        (self.models / "pitch.json").write_text(json.dumps({"mean_std_coef": [1] * len(voices)}))
        digests = {}
        for name in assets.VENDOR_SHA256:
            data = f"// fixture:{name}".encode()
            (self.vendor / name).write_bytes(data)
            digests[name] = hashlib.sha256(data).hexdigest()
        self.integrity = patch.dict(assets.VENDOR_SHA256, digests)
        self.integrity.start()
        self.addCleanup(self.integrity.stop)

    def test_single_file_roundtrip_and_notices(self):
        out = self.root / "nested/offline.html"
        expected = self.root / "expected.json"
        expected.write_text('[{"text":"sample","stressed":"sample"}]')
        build_html.build(self.models, self.vendor, out, expected)
        page = out.read_text(encoding="utf-8")
        payloads = dict(re.findall(r'data-name="([^"]+)">([^<]+)</script>', page))
        unpacked = {name: gzip.decompress(base64.b64decode(data)) for name, data in payloads.items()}
        self.assertEqual(unpacked["predictor"], (self.models / "predictor.onnx").read_bytes())
        self.assertEqual(unpacked["expected"], expected.read_bytes())
        self.assertEqual(len(json.loads(unpacked["voices"])), 29)
        self.assertEqual(json.loads(unpacked["tts-meta"])["sos"], "a")
        self.assertNotIn('src="', page)
        self.assertNotIn("<!-- SCRIPTS -->", page)
        self.assertIn("Copyright (c) Microsoft Corporation", page)
        self.assertIn("Copyright (c) 2025-present Silero Team", page)
        self.assertIn("Copyright (c) 2026 ArtemKo7v", page)

    def test_web_assets_resolve_and_rebuild_drops_optional_reference(self):
        out = self.root / "web"
        expected = self.root / "expected.json"
        expected.write_text("[]")
        build_web.build(self.models, self.vendor, out, expected)
        manifest = json.loads((out / "assets.json").read_text())
        self.assertIn("expected", manifest)
        for relative in manifest.values():
            self.assertTrue((out / relative).is_file(), relative)
        page = (out / "index.html").read_text(encoding="utf-8")
        for relative in re.findall(r'<script src="([^"]+)"', page):
            self.assertTrue((out / relative).is_file(), relative)
        self.assertIn("(() => {", (out / "src/tts.js").read_text())
        self.assertTrue((out / "licenses/silero-cis.txt").is_file())
        build_web.build(self.models, self.vendor, out)
        self.assertNotIn("expected", json.loads((out / "assets.json").read_text()))
        self.assertFalse((out / "models/expected.json").exists())

    def test_tampered_vendor_is_rejected(self):
        (self.vendor / "ort.wasm.min.js").write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            build_html.build(self.models, self.vendor, self.root / "bad.html")

    def test_incompatible_voice_metadata_is_rejected(self):
        (self.models / "meta.json").write_text(json.dumps(dict(
            symbols="a", symbol_to_id={}, speakers={}, sos="a", eos="b")))
        with self.assertRaisesRegex(ValueError, "speaker IDs"):
            build_web.build(self.models, self.vendor, self.root / "bad-web")


if __name__ == "__main__":
    unittest.main()
