---
license: apache-2.0
model_card_spec: "1.0"
pipeline_tag: tabular-regression
tags:
  - tabular-regression
  - tabular-foundation-model
  - in-context-learning
base_model: autogluon/mitra-regressor
---

# Mitra Regressor

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-autogluon%2Fmitra--regressor-ffcc4d?style=flat)](https://huggingface.co/autogluon/mitra-regressor)
[![GitHub](https://img.shields.io/badge/GitHub-autogluon%2Fautogluon-181717?style=flat&logo=github&logoColor=white)](https://github.com/autogluon/autogluon)
[![arXiv](https://img.shields.io/badge/arXiv-2510.21204-b31b1b.svg)](https://arxiv.org/abs/2510.21204)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)


###### Description

Mitra Regressor packages the `autogluon/mitra-regressor` checkpoint expected at Hugging Face revision `5f277aa8f69042d39d6ac3612aed18bb9279bd95`, a pretrained tabular foundation model developed by the AutoGluon team at Amazon Web Services for supervised regression on structured datasets. The model is a Transformer specialised for tables: it applies row-wise and column-wise attention so that relationships across observations and across features are both represented. It was pretrained across roughly 45 million synthetically generated datasets drawn from structural causal models and tree-based priors; the developers report that no real-world dataset was used directly in pretraining.

At inference the model conditions on the labelled training table as in-context support and emits one continuous estimate per query row; adaptation happens through in-context conditioning by default and, in this pipeline, through gradient fine-tuning when `fine_tune=true` (the DIMER default, `finetuner/train.py`). What this repository adds is the DIMER composition around those weights: a dataset validator (`validator/`), a fine-tuner that verifies the checkpoint digests before loading when the weights come from the Hub, enforces Mitra's row ceiling, seeds every RNG, scores a held-out split, and writes the result, provenance, and context artifacts DIMER consumes (`finetuner/`), plus the contract documents, sample datasets, and Colab tutorial. The upstream weights are not modified by this repository.

#### Intended Use and Limitations

The use cases below are the ones envisioned during development; the limits are the ones the code enforces.

###### Primary Intended Uses

Supervised prediction of a continuous numeric target from tabular features, where each observation is one row of numerical and categorical predictor columns. The pipeline takes a `train.csv` (optionally `val.csv`/`test.csv`) with a declared numeric target column and produces a fine-tuned or in-context Mitra predictor, one raw prediction per row, and holdout error metrics.

Concrete application domains envisioned during development: demand and quantity estimation, price or cost estimation, continuous risk or score prediction, scientific or engineering regression represented as feature tables, and resource-use or operational estimation formulated as row-wise tabular prediction — in particular the small-data regime, where the upstream authors report the model strongest (below roughly 5,000 samples and 100 features). The pipeline is meant to play the role of a strong zero-configuration baseline or a fine-tuned production model inside DIMER for tables that fit the ceilings: at most 10,000 training rows (`MITRA_ROW_LIMIT`; larger tables are sampled down) and at most 500 features (`MITRA_FEATURE_LIMIT`). The 10-class limit of the sibling classifier does not apply. Time-series, transactional, sensor, or panel data must first be represented as a supervised feature table; Mitra is not a sequence-forecasting model.

###### Primary Intended Users

Machine-learning researchers, data scientists, machine-learning engineers, and software developers working with structured datasets, and practitioners who want a pretrained foundation model for small-data tabular regression. The envisioned deployment setting is internal enterprise or research use through the DIMER platform — the fine-tuner and validator run as DIMER workers — not a public-facing service.

The pipeline assumes its users understand the provenance and semantics of their input data, the meaning and scale of the target variable, the consequences of prediction errors, and the limits of their own evaluation methodology: a user is expected to know that the served prediction is a raw, unclipped point estimate with no attached interval, that MAE on a 50-row holdout has wide variance, and that a sparse or heavy-tailed target needs a naive baseline for comparison. A user who cannot tell a held-out error from an in-sample one is outside the assumed competency.

###### Out-of-scope use cases

- **Capability boundaries:** categorical classification (the sibling `mitra-classifier-pipeline` does that); image, video, audio, natural-language, or other unstructured inputs; unsupervised clustering; causal-effect estimation; generative modelling; raw time-series forecasting without tabular feature construction; interval or quantile prediction (the pipeline emits a point estimate only).
- **Input boundaries:** a target column that yields no finite numeric values (validator check `target_is_numeric` refuses); more than 10,000 training rows (sampled down, never trained on in full); more than 500 features; fewer than 50 rows with a finite target (`MIN_TRAIN_ROWS`); archives whose members exceed 1 GiB uncompressed, 5,000,000 rows, or a 200× compression ratio (refused as zip-bomb guards).
- **Decision boundaries:** autonomous high-impact decisions — pricing of credit or insurance, clinical dosing, safety margins — without application-specific validation and a human decision-maker; treating published benchmark results as a guarantee on a new dataset.

#### Factors

Mitra's behaviour varies with the structure of the table it is given, not with a physical capture condition; the three subsections below say what that means for groups, instruments, and environment.

###### Groups

This pipeline is not human-centric by construction: Mitra was pretrained on synthetic datasets rather than any fixed human population, so no demographic group is an intrinsic development group of the foundation model, and the pretraining corpus is not group-audited because it contains no people. Demographic fairness or subgroup error parity has therefore **not** been established for the checkpoint, and the pipeline measures no subgroup metric.

Where the operator's downstream table describes people, the obligation transfers to the operator: identify the relevant groups in their own data, compute per-group MAE and RMSE on the holdout split, and check for disparate error before deployment — a regressor that is unbiased on average can still be systematically high for one group and low for another. The pipeline's provenance artifact records row counts and the split, not any demographic structure.

###### Instrumentation

Mitra consumes an abstract tabular representation rather than a raw sensor stream; the upstream pretraining did not depend on any real acquisition hardware. The instrument does not disappear because a table sits between it and the model: the operator's training and evaluation rows are produced by whatever systems fed the CSV — transactional databases, ETL pipelines, meters, sensors, survey instruments — and their characteristics (sampling rate, resolution, calibration, units, encoding of missing values) determine both feature quality and the scale of the target.

Instrument error reaches the model as feature or target error. Drift, miscalibration, a unit change, or a changed collection procedure between training and inference is not detectable by this pipeline; the validator checks that the target is numeric and finite and that the schema is consistent, not whether a column's meaning or unit has changed. Operators should document the instrumentation of downstream datasets separately.

###### Environment

**Operating environment.** The fine-tuner runs in the DIMER container on the `pytorch:2.8.0-cuda12.8` base image with AutoGluon 1.5.0; training expects a CUDA device, and the torch build is pinned by the image so that sm_120 (RTX 50-series) hosts keep the cu128 wheel. Precision follows AutoGluon's Mitra defaults. The validator is CPU-only. Fine-tuning under AutoGluon 1.5.0 is seeded (`_seed_everything`) but not guaranteed bit-deterministic.

**Data environment.** The reported error assumes the inference rows are drawn from the same distribution as the training table: same feature semantics, same units, same target range. Performance degrades, without warning from the pipeline, under geographic, institutional, temporal, or population shift, and with the technical factors that dominate tabular regression — dataset size, feature dimensionality and quality, target noise, heavy tails and outliers, sparse or intermittent targets, missing or erroneous values, categorical cardinality, preprocessing, leakage, and the fine-tuning configuration. Predictions outside the training target range are extrapolations the model has no basis for. Robustness to arbitrary distribution shift has not been established.

#### Metrics

Metrics are chosen for a point-estimate regressor whose intended use spans targets of very different scales and tail behaviour.

###### Performance Measures

The fine-tuner scores the trained predictor on the held-out split in `_score_holdout` (`finetuner/train.py`) and writes `mae` and `rmse`, computed directly from raw, unclipped predictions, alongside AutoGluon's full `predictor.evaluate()` output under `metrics.valEvaluation` with error metrics sign-flipped to their conventional lower-is-better form. The headline metric is the DIMER hyperparameter `eval_metric` (default `mean_absolute_error`), recorded as `headlineMetric`/`headlineScore`.

Why these: MAE is robust to outliers and interpretable in target units, which makes it the right default for a domain-agnostic pipeline whose targets range from counts to prices; RMSE weights large errors quadratically and is the informative one when a few big misses matter more than many small ones. Reading only MAE hides tail failures; reading only RMSE lets one outlier dominate. Both are therefore always written. R², MAPE, and scale-free measures appear in `valEvaluation` when AutoGluon returns them but are not headline metrics because R² misleads on low-variance targets and MAPE is undefined at zero. Upstream reports strong regression performance relative to TabPFNv2 and TabICL across TabRepo, TabZilla, AMLB, and TabArena, but publishes no single regression score; this pipeline claims none.

###### Decision thresholds

The pipeline applies no decision threshold and no clipping: `predictor.predict()` returns a raw continuous estimate and the served artifact returns it unchanged, so the reported error reflects what a caller will actually receive. No acceptance threshold on MAE or RMSE was set during development, because the pipeline is domain-agnostic and the tolerable error is a property of the deployment; published benchmark results are explicitly not production acceptance thresholds.

Any cutoff that turns a prediction into an action — a reorder point, a price band, a tolerance margin — is the deployment owner's to define and to calibrate on their own held-out residuals. Set it from the asymmetric cost of over- versus under-prediction: where an under-estimate is the expensive error, place the operating value above the point prediction by a margin derived from the holdout residual distribution, and revisit it whenever the input distribution or the target scale shifts.

###### Approaches to uncertainty and variability

The pipeline's reported metrics come from a single holdout split of the operator's table (requested size `validation_split`, effective size recorded), optionally capped at `DIMER_MAX_EVAL_ROWS` (default 50,000) rows. No dispersion is reported alongside the point value: one split, one run, no confidence interval. Operators who need one should repeat the run across seeds or use cross-validation on their own side.

Sources of run-to-run variability: the down-sampling when the table exceeds 10,000 rows, the holdout split, and gradient fine-tuning; all three are driven by the DIMER `seed` hyperparameter, which the fine-tuner propagates to Python, NumPy, and torch (`_seed_everything`). A fixed seed nonetheless does not guarantee bit-identical fine-tuning under AutoGluon 1.5.0 because of non-deterministic CUDA kernels. The pipeline emits no confidence output — no interval, no quantile, no predictive variance — so there is nothing to calibrate; a caller who needs an interval must estimate one from their own holdout residuals (for example, an empirical quantile of the absolute error) and should treat it as valid only within the training target range.

#### Ethical considerations and biases

No external ethics board reviewed this pipeline, and no clearance testing with a specific group took place; the subsections record what the developers considered and what the repository actually does.

###### Data

Mitra was pretrained exclusively on synthetic datasets, so the pretraining data do not consist of personally identifiable information, health, biometric, financial, or classified records — this is known from the upstream disclosure, which ends at the description of the synthetic priors; the generated tables themselves are not published. What this repository distributes: the DIMER worker code, contract documents, a tutorial, and small sample datasets under `examples/sample-data/` built by `examples/build_sample_datasets.py` and `examples/build_freshretailnet_dataset.py`; it does **not** distribute the checkpoint (the fine-tuner fetches it from the Hub and verifies the SHA-256 of `model.safetensors` and `config.json` against the pinned digests before loading, and `weights/` is gitignored).

Operators may fine-tune or evaluate Mitra on sensitive real-world tables. The pipeline does not audit the operator's data for personal, sensitive, or proprietary attributes — the validator checks structure and target type, not content — so the legality, privacy, consent, access control, and governance of downstream data remain with the application developer and data owner.

###### Human Life

The pipeline is not intended for decisions in health care, physical safety, criminal justice, legal rights, employment, credit, insurance, education access, or public benefits, and it has not been validated for any of them. The only validation performed is the contract testing in `scripts/` and the DIMER holdout scoring on the operator's own table; no clinical, regulatory, or independent domain validation has been carried out by the developers or by any external body, and upstream benchmark performance is not evidence of suitability.

Where such a use is foreseeable — a dosing or exposure regressor built on a clinical feature table, or a credit-limit regressor — it would be admissible only with independent domain validation on that operator's population, a human decision-maker between the prediction and the action, subgroup error evaluation, and whatever regulatory clearance the domain requires.

###### Mitigations

Implemented in this repository, each inspectable in the named code:

- **Supply-chain integrity:** the base model is `autogluon/mitra-regressor`, expected at revision `5f277aa8…`. Because AutoGluon 1.5.0's Mitra loader calls `hf_hub_download` without a revision argument, the enforceable guarantee is a digest check, not a revision pin: `resolve_and_verify_weights` in `finetuner/train.py` computes SHA-256 over the resolved `model.safetensors` and `config.json` and raises when either differs from `EXPECTED_WEIGHTS_SHA256` / `EXPECTED_CONFIG_SHA256`, recording the resolved commit and both digests in provenance with `enforced: true`. Weights uploaded through DIMER (`model_dir` set) are used verbatim and recorded with `enforced: false` — that path is deliberately not checked against the public digest, and the provenance says so. The torch/CUDA build is asserted by the Dockerfile.
- **Input integrity:** the validator resolves `train`/`val`/`test` deterministically and rejects nested or ambiguous archives, oversized members (> 1 GiB), tables over 5,000,000 rows, zip-bomb ratios (> 200×), a missing or dropped target column, a target with no finite numeric values (`target_is_numeric`), and fewer than 50 usable rows.
- **Statistical mitigations:** tables over 10,000 rows are sampled down with the seed before training; predictions are scored raw so that the reported error is the error a caller will see, not a clipped one.
- **Reproducibility:** `seed` propagates to Python, NumPy, and torch; the result artifact records the base revision, weight and config digests, AutoGluon version, effective split, and effective row counts.
- **Refusals:** the fine-tuner trains a single Mitra model with `fit_weighted_ensemble=False` and asserts that the requested model actually trained, so no silent fallback to another AutoGluon learner can occur; the pipeline emits no interval or quantile because it has no calibrated basis for one.

###### Risks and harms

- **Extrapolation outside the training range** (model-intrinsic): a query row beyond the support of the training table receives a point estimate with no signal of its own unreliability; borne by whoever the operator's decision affects; likely under normal use as the deployment drifts; magnitude set by what the estimate gates.
- **Amplification of input bias** (model-intrinsic): a table whose target encodes a historical disparity — past pay, past pricing — yields a regressor that reproduces it; borne by the data subjects in the disadvantaged group; realised whenever such a table is used without subgroup error evaluation.
- **Tail and sparsity failure** (model-intrinsic): heavy-tailed, intermittent, or zero-inflated targets can produce low MAE and large individual misses; borne by the operator who reads MAE alone.
- **Small-sample variance** (model-intrinsic): a holdout MAE on a few hundred rows can move materially between seeds; borne by the operator who ships on one split.
- **Automation bias** (use-context): a numerically precise estimate displaces human judgement; borne by the data subject; likely in any workflow that surfaces the number without its error.
- **Undetected leakage** (use-context): a feature derived from the target collapses the holdout error and fails in production; the validator does not detect it; borne by the operator and downstream users.
- **Benchmark over-generalisation** (use-context): reading upstream relative rankings as an expected error level on a new table; borne by whoever sets expectations from them.

###### Use cases

Distinct from the capability and decision boundaries listed under *Out-of-scope use cases*, the developers consider the following uses prohibited even where the model would produce a numerically plausible estimate:

- surveillance, biometric or demographic profiling, or social scoring of individuals;
- unlawful discrimination in employment, housing, credit, insurance, education, or healthcare access, including regression on a target that proxies a protected attribute (pay, premium, or limit set by group membership);
- deceptive, manipulative, or predatory applications, including exploitative price discrimination and presenting a point estimate as a certified measurement;
- clinical dosing, safety-margin, or legal-rights determinations without the validation and oversight described under *Human Life*;
- any use that violates the Apache-2.0 terms of the upstream `autogluon/mitra-regressor` weights and AutoGluon code, or the terms of the DIMER deployment.

---

## Model Details

- **Model name:** Mitra Regressor
- **Model identifier:** `autogluon/mitra-regressor`
- **Code repository:** [autogluon/autogluon](https://github.com/autogluon/autogluon)
- **Hugging Face repository:** [autogluon/mitra-regressor](https://huggingface.co/autogluon/mitra-regressor)
- **Developer:** AutoGluon team, Amazon Web Services (AWS)
- **Model family:** Tabular Foundation Model
- **Task:** Tabular Regression
- **Target type:** Continuous numeric value
- **Architecture:** Transformer with row-wise and column-wise attention
- **Transformer layers:** 12
- **Model / embedding dimension:** 512
- **Attention heads:** 4
- **Output dimension:** 1
- **Approximate parameter count:** ~72M in upstream descriptive material / ~75.7M in Hugging Face metadata
- **Pretraining:** Approximately 45 million synthetic datasets
- **Pretraining compute:** Eight NVIDIA A100 GPUs for approximately 60 hours
- **Real-world pretraining data:** None reported
- **License:** Apache License 2.0

## Checkpoint and Artifact Provenance

This card documents the following upstream Mitra Regressor checkpoint:

- **Hugging Face repository:** `autogluon/mitra-regressor`
- **Pinned revision:** `5f277aa8f69042d39d6ac3612aed18bb9279bd95`

### `model.safetensors`

- **Size:** 302,683,140 bytes
- **SHA-256:** `d8e75c62af0bec2fd404b0ad20a442d951d43ca6d331315cfcc0509b54f2c642`

### `config.json`

- **Size:** 81 bytes
- **SHA-256:** `2bc1ed5047f7c25368245e8ad32540a5fa28940b1ec05d3f1f454a09ff5384c1`

The exact raw content of `config.json` (81 bytes, single line without trailing newline) is:

```json
{"dim": 512, "dim_output": 1, "n_layers": 12, "n_heads": 4, "task": "REGRESSION"}
```

> **DIMER Model Hosting Note:** When retrieving Mitra model weights from DIMER, only the `model.safetensors` binary is hosted. Because AutoGluon requires `config.json` in the same directory to initialize the Transformer architecture, you can recreate `weights/config.json` alongside `model.safetensors` using the exact text above:
> 
> ```bash
> printf '{"dim": 512, "dim_output": 1, "n_layers": 12, "n_heads": 4, "task": "REGRESSION"}' > weights/config.json
> ```
> Or in PowerShell:
> ```powershell
> [System.IO.File]::WriteAllText("weights\config.json", '{"dim": 512, "dim_output": 1, "n_layers": 12, "n_heads": 4, "task": "REGRESSION"}', [System.Text.Encoding]::ASCII)
> ```

These parameters define the architecture into which the serialized weights are loaded. The `model.safetensors` file should therefore not be treated as fully self-describing in isolation. Correct reconstruction of this checkpoint requires the associated architecture configuration.

A change to `config.json` could alter how otherwise identical weight bytes are interpreted. For reproducible use, both the weight file and configuration should be verified against the revision and SHA-256 values above.

## Input

Mitra expects structured tabular data representing a supervised regression problem. Each dataset conceptually contains rows representing observations, numerical and/or categorical feature columns, and a continuous numeric target variable.

Input dimensionality and dataset size should remain within Mitra's supported regime, including a maximum of approximately 10,000 training samples and 500 features.

## Output

Mitra Regressor produces one continuous numeric prediction per input observation.

The associated `config.json` specifies `dim_output: 1`, reflecting the single scalar regression output. The semantic meaning and units of that output are determined by the downstream dataset.

AutoGluon's Mitra regression implementation min-max normalizes the target internally on the in-context support set, so downstream users do not generally need to manually scale the target solely for Mitra.

## Model Architecture

Mitra Regressor uses a Transformer architecture designed for tabular data. Its defining configuration is:

```json
{
  "dim": 512,
  "dim_output": 1,
  "n_layers": 12,
  "n_heads": 4,
  "task": "REGRESSION"
}
```

The architecture contains 12 Transformer layers, a 512-dimensional internal representation, four attention heads, a single regression output, and both row-wise and column-wise attention.

## Training Data

Mitra was pretrained on approximately **45 million synthetically generated tabular datasets**. The synthetic training distribution combines several families of priors, including structural causal models, gradient boosting, random forests, decision trees, and extra trees.

The developers report that **no real-world datasets were directly used during pretraining**.

A central design principle of Mitra is that the mixture of synthetic priors used during pretraining strongly influences transfer to real-world tabular problems. The prior mixture was selected based on standalone performance, diversity, and distinctiveness.

Pretraining used approximately **eight NVIDIA A100 GPUs for 60 hours**.

## In-Context Learning and Fine-Tuning

Mitra is fundamentally an **in-context learning tabular foundation model**. It can use labelled examples from a previously unseen tabular task as context when predicting values for new observations.

Mitra also supports **fine-tuning**, in which the pretrained parameters are adapted to a downstream dataset. Fine-tuning may improve performance depending on dataset characteristics and available compute.

Fine-tuned derivatives should be treated as application-specific model versions distinct from the upstream checkpoint documented here.

## Evaluation

The Mitra paper evaluates the model on established real-world tabular-learning benchmark collections, including **TabRepo**, **TabZilla**, **AutoML Benchmark (AMLB)**, and **TabArena**.

These datasets were used for evaluation rather than pretraining. The published results provide evidence that Mitra transfers effectively from synthetic priors to heterogeneous real tabular problems.

The authors report strong regression performance relative to contemporary tabular foundation models, including TabPFNv2 and TabICL, with improved sample efficiency in the evaluated regime. An important limitation reported in the associated documentation is that Mitra does **not consistently outperform TabPFNv2 on large-feature regression tasks**.

There is no single universal regression score that characterizes the foundation model across all datasets. Regression performance is dataset- and scale-dependent and should be evaluated with metrics appropriate to the downstream problem.

## Reproducibility

### Checkpoint Pinning

The documented upstream checkpoint is pinned to revision `5f277aa8f69042d39d6ac3612aed18bb9279bd95`.

Strict reproduction should verify both:

- `model.safetensors` SHA-256: `d8e75c62af0bec2fd404b0ad20a442d951d43ca6d331315cfcc0509b54f2c642`
- `config.json` SHA-256: `2bc1ed5047f7c25368245e8ad32540a5fa28940b1ec05d3f1f454a09ff5384c1`

Because `config.json` defines the architecture before the weights are loaded, matching the weight file alone is insufficient to establish complete model-version identity.

### AutoGluon Loader Limitation

AutoGluon 1.5.0's Mitra loader resolves the checkpoint using its Hugging Face repository identifier but does not expose a revision argument for directly pinning the underlying Hugging Face revision during normal model loading.

For strict reproduction, the exact resolved `model.safetensors` and `config.json` should therefore be verified against the revision and hashes recorded above.

### Random Seed Limitation

AutoGluon 1.5.0 does not fully enable Mitra's global `set_seed` behaviour. A fixed seed can make some stochastic components, including internal validation splitting, reproducible, but it should **not be assumed to guarantee complete bit-for-bit deterministic fine-tuning**.

Where reproducibility is important, users should record software versions, random seeds, data partitions, preprocessing, model and configuration hashes, fine-tuning parameters, and repeated-run variability.

## Limitations

Important limitations include:

1. Mitra is primarily designed for relatively small tabular datasets.
2. It supports a maximum of approximately 10,000 training samples.
3. It supports a maximum of approximately 500 features.
4. Performance depends strongly on the information contained in the input features and target structure.
5. Strong benchmark results do not guarantee strong performance on a particular downstream dataset.
6. Mitra does not consistently outperform TabPFNv2 on large-feature regression tasks.
7. Sparse, highly intermittent, or weak-signal targets may favor simpler baselines.
8. General demographic fairness has not been established.
9. Robustness to arbitrary distribution shift has not been established.
10. Domain-specific safety has not been established.
11. A fixed random seed does not guarantee completely deterministic fine-tuning under AutoGluon 1.5.0.
12. Exact checkpoint reproduction requires preserving both `model.safetensors` and `config.json`.

## License

Mitra Regressor is distributed under the **Apache License 2.0**. Apache-2.0 permits use, modification, redistribution, and hosted serving, including commercial use, subject to the license terms.

Redistributions should retain the applicable license and notices, and modifications should be documented as required by Apache-2.0. Licensing of downstream datasets and applications must be considered separately.

## Model Ownership and Attribution

Mitra Regressor was developed by the AutoGluon team at Amazon Web Services (AWS). Upstream source code is part of the AutoGluon project hosted at [autogluon/autogluon](https://github.com/autogluon/autogluon), and base model artifacts are distributed on Hugging Face at [autogluon/mitra-regressor](https://huggingface.co/autogluon/mitra-regressor). A downstream integration or fine-tuned derivative should distinguish the upstream foundation model from subsequent modifications and preserve applicable license and attribution information.

## Citation

Cite the original Mitra work, the AutoGluon framework, and the upstream repository:

### Papers

- **Mitra (2025):**  
  Zhang, X., Maddix, D. C., Yin, J., Erickson, N., Ansari, A. F., Han, B., Zhang, S., Akoglu, L., Faloutsos, C., Mahoney, M., Hu, T., Rangwala, H., Karypis, G., & Wang, Y. (2025). *Mitra: Mixed Synthetic Priors for Enhancing Tabular Foundation Models.* NeurIPS 2025. arXiv:2510.21204. https://doi.org/10.48550/arXiv.2510.21204

- **AutoGluon-Tabular (2020):**  
  Erickson, N., Mueller, J., Shirkov, A., Zhang, H., Larroy, P., Li, M., & Smola, A. (2020). *AutoGluon-Tabular: Robust and Accurate AutoML for Structured Data.* arXiv:2003.06505. https://doi.org/10.48550/arXiv.2003.06505

### Upstream Repository

- **AutoGluon Codebase:**  
  AutoGluon team, Amazon Web Services (AWS). *AutoGluon: AutoML for Image, Text, and Tabular Data* [Software]. GitHub. https://github.com/autogluon/autogluon

### BibTeX

```bibtex
@article{zhang2025mitra,
  title={Mitra: Mixed Synthetic Priors for Enhancing Tabular Foundation Models},
  author={Zhang, Xingjian and Maddix, Danielle C and Yin, Junwei and Erickson, Nick and Ansari, Abdul Fatir and Han, Boran and Zhang, Shenghao and Akoglu, Leman and Faloutsos, Christos and Mahoney, Michael and Hu, Tianpuxin and Rangwala, Huzefa and Karypis, George and Wang, Yuyang},
  journal={arXiv preprint arXiv:2510.21204},
  year={2025}
}

@article{erickson2020autogluon,
  title={AutoGluon-Tabular: Robust and Accurate AutoML for Structured Data},
  author={Erickson, Nick and Mueller, Jonas and Shirkov, Alexander and Zhang, Hang and Larroy, Pedro and Li, Mu and Smola, Alexander},
  journal={arXiv preprint arXiv:2003.06505},
  year={2020}
}

@misc{autogluon_repo,
  author = {Erickson, Nick and Mueller, Jonas and Shirkov, Alexander and Zhang, Hang and Larroy, Pedro and Li, Mu and Smola, Alexander and others},
  title = {AutoGluon: AutoML for Image, Text, and Tabular Data},
  year = {2020},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/autogluon/autogluon}}
}
```

## Evaluation Status

### Established by the Upstream Work

The upstream work establishes tabular regression capability, in-context learning, fine-tuning capability, synthetic-prior pretraining, evaluation across established real-world tabular benchmark suites, and strong performance within the evaluated small-data regime.

### Application-Dependent or Not Generally Established

The upstream evidence does not establish universal error on a particular downstream dataset, demographic fairness, subgroup parity, calibrated prediction intervals, adversarial robustness, robustness to arbitrary distribution shift, domain-specific safety, operational reliability, service-level guarantees, or suitability for high-impact decision-making.
