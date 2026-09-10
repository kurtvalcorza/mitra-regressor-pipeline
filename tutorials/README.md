# Mitra Regressor tutorials

[![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white)](https://github.com/kurtvalcorza/mitra-regressor-pipeline)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-autogluon%2Fmitra--regressor-ffcc4d?style=flat)](https://huggingface.co/autogluon/mitra-regressor)
[![arXiv](https://img.shields.io/badge/arXiv-2510.21204-b31b1b.svg)](https://arxiv.org/abs/2510.21204)

**DIMER Notebook Specification:** `1.0`

These notebooks are the user-facing tutorial surface for the Mitra Regressor pipeline. They exercise the repository-owned public API in `mitra_pipeline/` for core regression validation, Mitra fit/predict operations, and artifact-boundary verification.

| Notebook | Profile | Capability | Default runtime | Release status |
|---|---|---|---|---|
| [`mitra_regressor_colab.ipynb`](mitra_regressor_colab.ipynb) | `E2E` | acquire/verify model → validate data → evaluate/baselines → optional fine-tune → new-data inference → export → fresh reload | CPU; GPU only for optional fine-tuning | release-grade candidate; requires clean execution at release head |
| [`mitra_regressor_predictor_inference_colab.ipynb`](mitra_regressor_predictor_inference_colab.ipynb) | `ARTIFACT-INFERENCE` | externally supplied predictor ZIP → archive/provenance validation → serving reconstruction → new-data regression inference | CPU | release-grade candidate; requires clean execution at release head |

[Open E2E tutorial in Colab](https://colab.research.google.com/github/kurtvalcorza/mitra-regressor-pipeline/blob/main/tutorials/mitra_regressor_colab.ipynb)

[Open artifact-inference tutorial in Colab](https://colab.research.google.com/github/kurtvalcorza/mitra-regressor-pipeline/blob/main/tutorials/mitra_regressor_predictor_inference_colab.ipynb)

## Conformance contract

The E2E notebook:

- imports the repository's public `mitra_pipeline` API instead of reimplementing core fit/inference and artifact checks;
- installs pinned direct tutorial dependencies from [`requirements-colab.txt`](requirements-colab.txt);
- reports Python, AutoGluon, PyTorch/CUDA, repository commit, model id, and immutable model revision;
- supports a pinned upstream model path and a manifested DIMER offline-package path;
- validates exact SHA-256 values for `model.safetensors` and `config.json`;
- stages the verified files into an immutable offline Hugging Face snapshot;
- preserves the bundled sample's provided chronological train/validation/test partitions;
- offers a seeded random-holdout BYOD path only for approximately IID rows and a pre-split path for leakage-sensitive data;
- rejects duplicate raw CSV headers and validates finite numeric regression targets;
- reports missing-target drops, feature/row ceilings, and deterministic row capping;
- detects exact record overlap across supplied splits;
- distinguishes in-context support from gradient fine-tuning;
- reports MAE/RMSE/R² and computes mean/median `DummyRegressor`, LightGBM, and Random Forest baselines on the same partitions;
- keeps independent test evidence out of model selection;
- provides a separate, gated new-data inference path;
- exports machine-readable metrics and provenance;
- packages the actual AutoGluon predictor, including the preprocessing/support state required for inference;
- writes `artifact_manifest.json` with every artifact file's size and SHA-256; and
- reloads from the serialized ZIP in a fresh directory, validates the manifest/provenance before deserialization, and checks prediction equivalence with an explicit tolerance.

The artifact-inference notebook:

- requires an artifact supplied from outside its own execution;
- requires a trusted whole-archive SHA-256 by default before Python deserialization, with only an explicit expert override for already-trusted local artifacts;
- rejects absolute paths, traversal, backslash paths, symlinks, suspicious compression ratios, oversized members, and oversized total expansion;
- requires `artifact_manifest.json` and `tutorial_run_metadata.json`;
- verifies artifact format/version, model/revision, problem type, complete file inventory, sizes, and digests before `TabularPredictor.load(...)`;
- refuses incompatible AutoGluon or Python major/minor versions;
- restores the predictor's fitted preprocessing/support state from the artifact;
- validates a genuinely new CSV and writes `predictions.csv`; and
- explicitly identifies regression output as point predictions without calibrated per-row uncertainty.

## Model provenance

- Model: `autogluon/mitra-regressor`
- Task: tabular regression
- AutoGluon: `1.5.0`
- Immutable upstream revision: `5f277aa8f69042d39d6ac3612aed18bb9279bd95`
- Weights SHA-256: `d8e75c62af0bec2fd404b0ad20a442d951d43ca6d331315cfcc0509b54f2c642`
- Config SHA-256: `2bc1ed5047f7c25368245e8ad32540a5fa28940b1ec05d3f1f454a09ff5384c1`
- Supported ceiling: 10,000 training rows and 500 features
- Output: one continuous point prediction per row; no calibrated per-prediction interval

The model repository supplies weights/configuration only; Mitra's executable integration comes from the pinned AutoGluon package. The notebooks do not execute Python code from the model repository.

## Data and evaluation

The default sample is `freshretailnet-h7.zip`, pinned to repository revision `5625a9eeca94b8c72b9ad1ec78d07ecbaa720903`. It is derived from FreshRetailNet-50K and is used for tutorial/sanity evidence, not benchmarking.

The provided sample partition membership is preserved. The literal default CPU tutorial deterministically uses 512 training rows and 256 rows from each evaluation partition as a bounded smoke subset; users can raise the notebook form controls to use more or all rows from the pinned convenience sample. This smoke-only cap does not apply to BYOD. A BYOD single-CSV path uses a deterministic random holdout and explicitly assumes approximately IID rows. Time-dependent, grouped, panel, embargoed, patient/device-level, spatial, or otherwise leakage-sensitive workflows should provide pre-split train/validation/test files.

Regression evaluation includes executable constant baselines rather than quoting fixed development-run numbers. MAE and RMSE remain in target units; R² is complementary. Sample values must not be generalized to other datasets.

## Artifact contract

The E2E tutorial exports `mitra-predictor.zip`. The archive contains the selected AutoGluon predictor plus:

- `tutorial_run_metadata.json` — model/revision, runtime, data identity, selected variant, adaptation configuration, metrics, feature order, target, and provenance;
- `artifact_manifest.json` — format/version and complete file inventory with size and SHA-256 for every other artifact file.

Because Mitra inference uses support/training context and AutoGluon preprocessing state, the exported predictor may contain or encode information derived from source data. Apply the source dataset's confidentiality, licensing, disclosure, and retention rules to the artifact.

Path safety and manifest consistency do not make Python serialization safe. Only load predictor ZIPs from trusted producers.

The optional DIMER offline model ZIP has its own normative manifest contract documented in [`DIMER_MODEL_PACKAGE.md`](DIMER_MODEL_PACKAGE.md).

## Release verification

Static checks run in ordinary CI:

- notebook JSON/metadata checks;
- Python-cell compilation after notebook-magics are stripped;
- conformance markers and absence of placeholder text;
- CSV/header regression checks;
- public API archive/manifest unit tests.

A separate **Notebook release execution** workflow executes the current E2E default path in a clean hosted runner, produces a predictor artifact, derives new inference rows, and executes the `ARTIFACT-INFERENCE` notebook against that externally produced artifact. A release claim must cite the successful workflow/PR head; static CI alone is not execution evidence.

## AI use and provenance

These tutorials have been developed with substantial AI assistance under human direction and review. AI attribution is authorship provenance, not sign-off. Clean execution, static checks, and human review remain the evidence for a release.

## Sample portfolio

The E2E notebook preserves the original FreshRetailNet temporal-demand sample and also exposes the sample portfolio from PR #19: Insurance Medical Charges (`charges`) and Ames Housing (`SalePrice`). These archives and their provenance/license details live under `examples/sample-data/`. The older `Sample dataset (FreshRetailNet)` selector remains accepted for backward compatibility. Sample metrics remain tutorial/sanity evidence, not benchmark claims.
