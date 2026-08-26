---
license: apache-2.0
pipeline_tag: tabular-regression
tags:
  - tabular-regression
  - tabular-foundation-model
  - in-context-learning
base_model: autogluon/mitra-regressor
---

# Mitra Regressor

## Description

Mitra Regressor is a pretrained tabular foundation model developed by the AutoGluon team at Amazon Web Services (AWS) for supervised regression on structured or tabular datasets.

The model predicts a continuous numeric target from numerical and categorical input features. Mitra uses a Transformer architecture specialized for tabular data, including both row-wise and column-wise attention so that it can model relationships across observations and features.

Unlike conventional regression models trained directly on one application dataset, Mitra was pretrained across approximately 45 million synthetically generated tabular datasets. Its synthetic pretraining distribution combines structural causal models with several tree-based prior families, including gradient boosting, random forests, decision trees, and extra trees. The developers report that no real-world datasets were used directly during pretraining.

Mitra operates as an in-context learning tabular foundation model and additionally supports fine-tuning on downstream datasets.

## Model Details

- **Model name:** Mitra Regressor
- **Model identifier:** `autogluon/mitra-regressor`
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

The associated configuration is:

```json
{
  "dim": 512,
  "dim_output": 1,
  "n_layers": 12,
  "n_heads": 4,
  "task": "REGRESSION"
}
```

These parameters define the architecture into which the serialized weights are loaded. The `model.safetensors` file should therefore not be treated as fully self-describing in isolation. Correct reconstruction of this checkpoint requires the associated architecture configuration.

A change to `config.json` could alter how otherwise identical weight bytes are interpreted. For reproducible use, both the weight file and configuration should be verified against the revision and SHA-256 values above.

## Intended Use and Limitations

### Primary Intended Uses

Mitra Regressor is intended for supervised prediction of continuous numeric targets from structured or tabular features.

Appropriate applications include:

- demand and quantity estimation;
- price or cost estimation;
- continuous risk or score prediction;
- scientific or engineering regression represented as feature tables;
- resource-use or operational forecasting formulated as row-wise tabular prediction; and
- other supervised regression problems involving relatively small tabular datasets.

Mitra is particularly targeted at the small-data regime and is reported to be strongest below approximately 5,000 samples and 100 features.

Its supported upper limits include:

- **10,000 training samples**
- **500 features**

The 10-class limit associated with Mitra Classifier does not apply to this regression checkpoint.

For time-series, transactional, sensor, or panel datasets, the source data must first be represented as an appropriate supervised feature table. Mitra Regressor is not itself a general-purpose sequence forecasting model.

### Primary Intended Users

Mitra Regressor is intended primarily for machine-learning researchers, data scientists, machine-learning engineers, software developers, researchers working with structured datasets, and practitioners seeking a pretrained foundation model for small-data tabular regression.

Users should understand the provenance and semantics of their input data, the meaning and scale of the target variable, the consequences of prediction errors, and the limitations of their evaluation methodology.

### Out-of-Scope Use Cases

Mitra Regressor is not intended for:

- categorical classification; a separate Mitra classifier checkpoint is available;
- image, video, audio, natural-language, or other unstructured-data tasks;
- datasets exceeding the model's supported sample or feature limits;
- unsupervised clustering;
- causal-effect estimation;
- generative modelling;
- direct raw time-series forecasting without tabular feature construction; or
- autonomous high-impact decision-making without application-specific validation and appropriate oversight.

Published benchmark performance should not be interpreted as a guarantee of performance on a new dataset.

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

## Performance Measures

Appropriate downstream regression metrics may include:

- mean absolute error (MAE);
- root mean squared error (RMSE);
- mean squared error (MSE);
- coefficient of determination (R²);
- mean absolute percentage error (MAPE) where mathematically appropriate; and
- task-specific normalized or scale-independent error measures.

Metric selection should reflect the target distribution and consequences of error. MAE is robust and interpretable in target units; RMSE emphasizes large errors; R² measures variance explained but can be misleading when used without absolute error metrics.

## Decision Thresholds

Regression does not define a universal class decision threshold. Any operational cutoff applied to a numeric prediction must be defined by the downstream application and should be documented separately from the foundation model.

## Factors

### Groups

Mitra was pretrained using synthetic datasets rather than datasets representing a fixed human population. No demographic groups are therefore intrinsic development groups of the foundation model.

Where Mitra is used on human-related datasets, relevant groups and subgroup error should be identified and evaluated for the particular downstream application.

### Instrumentation

Mitra consumes structured tabular features rather than raw signals from a specific physical instrument. For downstream datasets derived from physical measurements, the instrumentation used to produce those features should be documented separately.

### Environment

Mitra was not developed for one physical environment. Environmental conditions become relevant when they influence input variables or the statistical distribution of downstream data.

### Technical Factors

Performance may be affected by dataset size, feature dimensionality, target distribution, missing values, erroneous observations, feature quality, target noise, heavy tails, outliers, categorical cardinality, preprocessing, leakage, random variation, fine-tuning configuration, and distribution shift.

Sparse or highly intermittent regression targets can be particularly challenging. Strong naive baselines should be included when appropriate.

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

## Approaches to Uncertainty and Variability

For downstream regression, appropriate uncertainty assessment may include repeated experiments across seeds, confidence intervals for aggregate error, cross-validation where appropriate, independent holdout evaluation, temporal validation, external validation, subgroup error analysis, residual diagnostics, and prediction-interval methods where supported by the surrounding application.

## Ethical Considerations and Biases

### Data

Mitra was pretrained exclusively on synthetic datasets rather than a corpus of real-world human records. This does not remove privacy, fairness, or governance risks from downstream applications using sensitive real-world data.

### Human Life

Mitra is a general-purpose tabular foundation model and was not specifically developed or validated for autonomous decisions concerning health care, physical safety, criminal justice, legal rights, employment, credit, insurance, education access, public benefits, or other high-impact matters affecting human welfare.

### Mitigations

Appropriate downstream mitigations include dataset provenance checks, data-quality validation, leakage prevention, comparison against strong baselines, subgroup evaluation, robust error metrics, distribution-shift assessment, independent testing, human review for material decisions, and post-deployment monitoring.

### Risks and Harms

Potential risks include inaccurate numeric predictions, dataset bias, unequal error across subgroups, distribution shift, leakage, sensitivity to outliers or target sparsity, automation bias, and overgeneralization from benchmark results.

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

Mitra Regressor was developed by the AutoGluon team at Amazon Web Services (AWS). A downstream integration or fine-tuned derivative should distinguish the upstream foundation model from subsequent modifications and preserve applicable license and attribution information.

## Citation

Zhang, X., Maddix, D. C., Yin, J., Erickson, N., Ansari, A. F., Han, B., Zhang, S., Akoglu, L., Faloutsos, C., Mahoney, M., Hu, T., Rangwala, H., Karypis, G., & Wang, Y. (2025). *Mitra: Mixed Synthetic Priors for Enhancing Tabular Foundation Models.* NeurIPS 2025. arXiv:2510.21204. https://doi.org/10.48550/arXiv.2510.21204

## Evaluation Status

### Established by the Upstream Work

The upstream work establishes tabular regression capability, in-context learning, fine-tuning capability, synthetic-prior pretraining, evaluation across established real-world tabular benchmark suites, and strong performance within the evaluated small-data regime.

### Application-Dependent or Not Generally Established

The upstream evidence does not establish universal error on a particular downstream dataset, demographic fairness, subgroup parity, calibrated prediction intervals, adversarial robustness, robustness to arbitrary distribution shift, domain-specific safety, operational reliability, service-level guarantees, or suitability for high-impact decision-making.
