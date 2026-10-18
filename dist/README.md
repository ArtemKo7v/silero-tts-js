# Build outputs

- `web/`: generated HTTP demo; serve with `python scripts/serve.py`.
- `silero.html`: self-contained offline page; open directly from disk.
- `silero-test.html`: optional standalone build with a Python reference corpus.

Generated files are ignored by Git. Rebuild with the scripts documented in the
[README](../README.md). Distribute large HTML files and model bundles as release
assets rather than adding them to repository history.
