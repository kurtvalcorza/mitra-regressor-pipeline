---
license: apache-2.0
pipeline_tag: tabular-regression
tags:
  - tabular-regression
  - tabular-foundation-model
  - in-context-learning
base_model: autogluon/mitra-regressor
---

# Mitra regressor — model card (for the DIMER pipeline)

**This is not a model this project trained.** Mitra was pretrained by the AutoGluon team at
AWS and is used here unmodified under Apache-2.0. This card records what the pipeline uses,
where it comes from, and how it behaves; see the [README](README.md) for the pipeline itself.

## Summary

| | |
|---|---|
| Base model | [`autogluon/mitra-regressor`](https://huggingface.co/autogluon/mitra-regressor) |
| Pinned revision | `5f277aa8f69042d39d6ac3612aed18bb9279bd95` |
| Architecture | 12-layer Transformer, 512 embedding, 4 heads, row + column (2D) attention |
| Parameters | ~72M (upstream card) / 75.7M (HF metadata) |
| Pretraining | 45M synthetic datasets on 8×A100 (~60 h); no real data seen |
| Task | tabular regression (numeric target) |
| Licence | Apache-2.0 — redistribution and hosted serving permitted, including commercial |

`model.safetensors` is 302,683,140 bytes, SHA-256
`d8e75c62af0bec2fd404b0ad20a442d951d43ca6d331315cfcc0509b54f2c642`; `config.json` is 81 bytes,
SHA-256 `2bc1ed5047f7c25368245e8ad32540a5fa28940b1ec05d3f1f454a09ff5384c1`.

## Providing the weights to DIMER

The weights are **not committed to this repository** (289 MB; and DIMER's build may not fetch
Git LFS). A local copy is kept in `weights/` (gitignored — never pushed) for convenience: pick
`weights/model.safetensors` in the wizard's upload box, or use it for a local image bake.
Provide them to the pipeline in one of three ways:

1. **HuggingFace Model ID (default).** Enter `autogluon/mitra-regressor` as the Base Model;
   AutoGluon downloads it at runtime. Requires egress to `huggingface.co`.
2. **Upload weights (no egress).** Mount an uploaded checkpoint directory (holding
   `model.safetensors` and `config.json`) and set `DIMER_MODEL_DIR` to it. The fine-tuner
   installs those exact bytes into the loader's cache and uses them verbatim — no egress.
3. **Bake into the image.** Uncomment Option A in the fine-tuner `Dockerfile` to download the
   pinned revision at build time (needs egress at build), or `COPY` a local copy in.

Fetch the pinned bytes with the Hugging Face CLI:

```bash
hf download autogluon/mitra-regressor model.safetensors config.json \
  --revision 5f277aa8f69042d39d6ac3612aed18bb9279bd95 --local-dir .
```

**Weight verification.** AutoGluon 1.5.0's Mitra loader resolves a checkpoint by Hugging Face
repo id and does not accept a revision argument, so the base weights cannot be pinned by
revision through it. Instead, before fitting, the fine-tuner resolves the exact
`model.safetensors` **and `config.json`** the loader will use and **verifies both SHA-256s
against the expected values above** (config.json carries the architecture Mitra builds before
the weights load, so a drifted config with matching weights would still change the model — it is
pinned too); a mismatch on either fails the run. `result.json`'s `provenance` block records the
resolved revision, the loaded `weightsSha256`/`configSha256`, and whether the check was enforced.
Baking the pinned revision (option 3) makes the loaded bytes deterministic; option 2
(`DIMER_MODEL_DIR`) records the uploaded bytes' checksums but does not compare them to the public
pinned values.

## How the pipeline uses it

- **Fine-tune (GPU)** — adapts the pretrained weights to the uploaded table. Requires a GPU.
- **Zero-shot (CPU)** — in-context inference with no weight update; used automatically when no
  GPU is available (Mitra's fine-tuning backward pass is unsupported on many CPUs).

The fine-tuner selects the mode from GPU availability at runtime and records it in
`result.json` (`metrics.mode`, `metrics.device`). See the [README](README.md).

The run seed and — when it maps to one of Mitra's native metrics — the eval metric are passed
into Mitra itself (recorded as `metrics.mitraSeed` and `metrics.mitraMetric`): the seed drives
Mitra's internal validation split, and the metric drives fine-tune early stopping. Note that
AutoGluon 1.5.0 disables Mitra's global `set_seed` (an upstream FIXME), so a fixed seed makes
the internal split reproducible but not the entire fit.

## Applicability and limits

Mitra is strongest on **small tabular data** (below ~5,000 samples and ~100 features). Hard
limits: **10,000 training rows, 500 features, 10 classes** (classification). It needs about
**~10 GB of memory** (measured on the ~4,200-row sample; it grows with rows and features);
request a profile that clears that with headroom (see the README's resource profile).

## Measured behaviour in this pipeline

Small smoke-test runs on the bundled sample (FreshRetailNet demand 7 days ahead; 4,180 train /
1,600 val / 1,600 test rows, one seed, a **purged per-series chronological split** with a
7-row embargo at each boundary). Holdout MAE (lower is better), against a roll-7-mean naive
forecast:

| Mode | Val MAE | Test MAE | Roll-7 naive (val / test) |
|---|---|---|---|
| fine-tune (GPU) | **0.369** | **0.429** | 0.389 / 0.438 |
| zero-shot (CPU) | 0.413 | 0.436 | 0.389 / 0.438 |

On this leak-free forward-looking split, fine-tuning Mitra **modestly beats** the roll-7-mean
baseline on both val and test; zero-shot is about even with it. The margin is small — for a
plain short-horizon forecast a lag/rolling baseline is a strong reference, and a foundation
model does not win by default. Mitra's advantage grows on tabular problems where the features
carry signal a naive rule misses. These are direction, not scores; treat Mitra's published
benchmarks as evidence of strong performance where signal exists, not a guarantee here.

## Licence

Apache-2.0. Redistribution and hosted serving are both permitted, including commercial use, on
the conditions of retaining the `LICENSE` and stating modifications (there are none). The
Apache-2.0 licence text ships with the weights upstream.

## Citation

Cite the original work, not this repository:

> Zhang, X., Maddix, D. C., Yin, J., Erickson, N., Ansari, A. F., Han, B., Zhang, S., Akoglu,
> L., Faloutsos, C., Mahoney, M., Hu, T., Rangwala, H., Karypis, G., & Wang, Y. (2025).
> *Mitra: Mixed Synthetic Priors for Enhancing Tabular Foundation Models.* NeurIPS 2025.
> arXiv:2510.21204. https://doi.org/10.48550/arXiv.2510.21204
