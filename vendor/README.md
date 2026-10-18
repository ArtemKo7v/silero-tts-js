# Local runtime files

Run `python scripts/fetch_vendor.py` from the repository root to download:

- `ort.wasm.min.js`
- `ort-wasm-simd-threaded.mjs`
- `ort-wasm-simd-threaded.wasm`

All three come from the pinned `onnxruntime-web@1.20.1/dist` package on jsDelivr.
The downloader and both builders verify the SHA-256 digests in `scripts/assets.py`.
Downloaded binaries are ignored by Git. See the original
[MIT license](../licenses/onnxruntime.txt).
