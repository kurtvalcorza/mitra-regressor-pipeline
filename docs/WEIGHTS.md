# Weight provenance and local snapshot layout

| Field | Value |
|---|---|
| Upstream model | `autogluon/mitra-regressor` (Hugging Face) |
| Files | `model.safetensors` (302,683,140 bytes), `config.json` (81 bytes) |
| Immutable revision | `5f277aa8f69042d39d6ac3612aed18bb9279bd95` |
| SHA-256 `model.safetensors` | `d8e75c62af0bec2fd404b0ad20a442d951d43ca6d331315cfcc0509b54f2c642` |
| SHA-256 `config.json` | `2bc1ed5047f7c25368245e8ad32540a5fa28940b1ec05d3f1f454a09ff5384c1` |
| License | Apache-2.0 (upstream checkpoint; `LICENSE` ships next to the weights) |
| Package constants | `MODEL_ID`, `MODEL_REVISION` (alias `PINNED_REVISION`), `MODEL_LICENSE`, `MODEL_KEY = "mitra-regressor"`, `WEIGHTS_SHA256`, `CONFIG_SHA256` in `mitra_pipeline/tutorial_api.py` |

## Local snapshot (fleet scheme, NOTEBOOK_SPEC 1.1 ST3/ST4)

The pinned snapshot lives in `weights/mitra-regressor/` (the `MODEL_KEY`). The committed `dimer-base-manifest.json`
there lists both files' paths, byte sizes and SHA-256 and is the parity anchor the standalone tutorials carry inline;
`model.safetensors` is git-ignored (`weights/**/*.safetensors`) while `config.json`, `LICENSE` and the manifest are
committable. `verify_snapshot` re-hashes every manifest entry and asserts the manifest digests equal the package's
`WEIGHTS_SHA256` / `CONFIG_SHA256`; `stage_missing_files(allow_download=True)` fetches only the entries that are
absent, from the Hub at the immutable revision; `MitraRegressionPipeline.from_pretrained(weights_dir=...)` runs both
and then calls the existing `stage_verified_hf_snapshot`, which copies the verified bytes into an offline Hugging Face
cache layout (`HF_HUB_OFFLINE=1`) so AutoGluon's Mitra resolves exactly the pinned revision without a network path. To
use an offline copy (for example the DIMER wizard's model ZIP), place both files in that directory before running.

```bash
python -c "from mitra_pipeline import stage_missing_files, verify_snapshot; print(stage_missing_files(allow_download=True)); print(verify_snapshot()['files'])"
```

The digest checks establish byte integrity against this repository's pinned identity, not producer authenticity or
model quality. The DIMER fine-tuner image bakes the same files at build time (see `DEPLOYMENT.md`).
