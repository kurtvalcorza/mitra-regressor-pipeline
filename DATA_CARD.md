# Data Card

## Version Details

### Input

Mitra Regressor expects **structured tabular data** representing a supervised regression problem.

Each dataset consists conceptually of:

- **Rows:** individual observations or examples.
- **Feature columns:** numerical and/or categorical predictor variables.
- **Target column:** a continuous numeric variable to be predicted.

The model is designed primarily for relatively small tabular datasets. Its supported regime includes:

- maximum training samples: **10,000**;
- maximum features: **500**.

Mitra is reported to perform particularly well in the small-data regime, especially on datasets with approximately **5,000 or fewer samples and 100 or fewer features**.

Mitra is not intended to consume raw images, text, audio, video, or other unstructured data directly. Such information must first be transformed into an appropriate tabular feature representation.

### Output

Mitra Regressor produces **one continuous numeric prediction per input observation**.

The semantic meaning and units of the output are determined by the downstream dataset. Examples include demand, quantity, price, cost, score, measurement, or another continuous target.

The associated architecture configuration specifies `dim_output: 1`, corresponding to a single scalar regression output.

AutoGluon's Mitra regression implementation min-max normalizes the target internally on the in-context support set, so users generally do not need to manually scale the target solely for Mitra.

### Type

Mitra Regressor is a **Transformer-based tabular foundation model** developed by the AutoGluon team at Amazon Web Services (AWS).

It uses row-wise and column-wise attention to model relationships across both observations and features.

The accompanying upstream `config.json` defines the regressor architecture as:

```json
{
  "dim": 512,
  "dim_output": 1,
  "n_layers": 12,
  "n_heads": 4,
  "task": "REGRESSION"
}
```

The configuration specifies:

- `dim: 512` — internal model or embedding dimension;
- `dim_output: 1` — a single continuous regression output;
- `n_layers: 12` — number of Transformer layers;
- `n_heads: 4` — number of attention heads;
- `task: "REGRESSION"` — identifies the checkpoint as the regression variant of Mitra.

The model is described upstream as approximately **72 million parameters**, while Hugging Face metadata reports approximately **75.7 million parameters**.

Mitra is an **in-context learning tabular foundation model** and also supports fine-tuning on a downstream dataset.

### Paper or Other Resource for Information

Primary publication:

**Mitra: Mixed Synthetic Priors for Enhancing Tabular Foundation Models**

Xiyuan Zhang, Danielle C. Maddix, Junming Yin, Nick Erickson, Abdul Fatir Ansari, Boran Han, Shuai Zhang, Leman Akoglu, Christos Faloutsos, Michael W. Mahoney, Cuixiong Hu, Huzefa Rangwala, George Karypis, and Bernie Wang.

NeurIPS 2025.

arXiv:2510.21204  
DOI: 10.48550/arXiv.2510.21204

Additional resources include the Hugging Face repository `autogluon/mitra-regressor`, AutoGluon documentation and source code, Amazon Science materials, and the Mitra research paper and associated benchmark resources.

### Citation Details

Zhang, X., Maddix, D. C., Yin, J., Erickson, N., Ansari, A. F., Han, B., Zhang, S., Akoglu, L., Faloutsos, C., Mahoney, M., Hu, T., Rangwala, H., Karypis, G., & Wang, Y. (2025). *Mitra: Mixed Synthetic Priors for Enhancing Tabular Foundation Models.* Advances in Neural Information Processing Systems (NeurIPS 2025). arXiv:2510.21204. https://doi.org/10.48550/arXiv.2510.21204

### Other Relevant Information

- **Developer:** AutoGluon team, Amazon Web Services (AWS)
- **Model identifier:** `autogluon/mitra-regressor`
- **Model family:** Tabular Foundation Model
- **Task:** Tabular Regression
- **Target type:** Continuous numeric value
- **License:** Apache License 2.0
- **Pretraining data:** Approximately 45 million synthetic tabular datasets
- **Pretraining compute:** Eight NVIDIA A100 GPUs for approximately 60 hours
- **Real-world pretraining data:** None reported
- **Pinned revision:** `5f277aa8f69042d39d6ac3612aed18bb9279bd95`

#### Associated Model Artifacts

`model.safetensors`

- Size: **302,683,140 bytes**
- SHA-256: `d8e75c62af0bec2fd404b0ad20a442d951d43ca6d331315cfcc0509b54f2c642`

`config.json`

- Size: **81 bytes**
- SHA-256: `2bc1ed5047f7c25368245e8ad32540a5fa28940b1ec05d3f1f454a09ff5384c1`

The model version represented in this card is associated with the following configuration:

```json
{
  "dim": 512,
  "dim_output": 1,
  "n_layers": 12,
  "n_heads": 4,
  "task": "REGRESSION"
}
```

The registry may store only `model.safetensors`, but the configuration is part of the model-version definition and should be preserved together with the weights. The weights file is not fully self-describing in isolation.

---

# 1. Evaluation Datasets

## i. Dataset

The Mitra paper evaluates the model using established real-world tabular-learning benchmark collections, including:

- **TabRepo**
- **TabZilla**
- **AutoML Benchmark (AMLB)**
- **TabArena**

These collections contain heterogeneous tabular regression and classification problems spanning different application domains, dataset sizes, feature structures, and statistical properties.

The real-world benchmark datasets were used for **evaluation**, not for Mitra's synthetic pretraining.

## ii. Motivation

Multiple benchmark collections were used because performance on one or a few datasets would provide limited evidence about the generalization ability of a tabular foundation model.

Regression datasets can differ substantially in sample size, feature dimensionality, numeric and categorical composition, target scale, target skew, outliers, missingness, statistical relationships, and application domain.

Using multiple established benchmark collections tests whether inductive biases learned from synthetic priors transfer to previously unseen real tabular data and permits comparison with contemporary tabular foundation models and conventional machine-learning approaches.

## iii. Preprocessing

Evaluation follows the tabular-processing and benchmark procedures described in the Mitra publication and associated AutoGluon implementation.

Individual datasets are transformed into the representation required by Mitra while preserving the evaluation structure of the relevant benchmark. Mitra supports mixed numerical and categorical tabular features.

For regression, the Mitra implementation min-max normalizes the target on the in-context support set internally. Manual target scaling is therefore not generally required solely for model compatibility.

Exact preprocessing, splits, normalization conventions, and benchmark protocols differ across benchmark suites. The Mitra paper and corresponding benchmark documentation should be consulted when exact reproduction of a reported result is required.

---

# 2. Training Datasets

## i. Dataset

Mitra was pretrained on approximately **45 million synthetically generated tabular datasets** rather than a fixed corpus of real-world datasets.

The synthetic pretraining mixture incorporates several families of priors, including:

- structural causal models (SCMs);
- gradient-boosting-based priors;
- random-forest-based priors;
- decision-tree-based priors; and
- extra-trees-based priors.

The developers report that **no real-world datasets were directly used during Mitra pretraining**.

## ii. Motivation

A central design principle of Mitra is that the choice and mixture of synthetic priors substantially influence how effectively a tabular foundation model generalizes to real-world problems.

Synthetic generation allows the model to experience a very large and diverse collection of learning problems without requiring a correspondingly massive corpus of real tabular datasets.

The authors select and combine priors according to three principal considerations:

1. **Standalone performance** — whether a prior produces transferable behaviour that performs well on real tabular datasets.
2. **Diversity** — whether the prior adds substantially different statistical structures to the pretraining distribution.
3. **Distinctiveness** — whether the prior contributes useful behaviour not already represented by other priors in the mixture.

Combining structural causal and tree-based priors exposes the model to different functional relationships and inductive biases.

## iii. Preprocessing

Because Mitra's pretraining data are synthetically generated, preprocessing is integrated into the synthetic-data generation and model-training procedure rather than consisting of cleaning a fixed real-world corpus.

Synthetic datasets are generated according to the selected prior families and transformed into the structured representation consumed by the Transformer.

Detailed prior distributions, synthetic-generation procedures, sampling strategies, and training methodology are described in the Mitra paper and associated implementation.

Pretraining used approximately **45 million synthetic datasets** and was conducted using **eight NVIDIA A100 GPUs for approximately 60 hours**.

---

# Quantitative Analyses

## Unitary Results

The Mitra paper evaluates regression performance across heterogeneous real-world tabular benchmarks rather than assigning one universal error value to the foundation model.

The authors report strong regression performance relative to contemporary tabular foundation models, including TabPFNv2 and TabICL, with improved sample efficiency in the evaluated small-data regime.

An important qualification is that Mitra does **not consistently outperform TabPFNv2 on large-feature regression tasks**. This indicates that relative performance depends on dataset characteristics, particularly feature dimensionality and problem structure.

Unlike classification accuracy, regression metrics are scale-dependent. A single MAE, RMSE, or MSE value aggregated across datasets with different target units is generally not interpretable as an intrinsic model score unless normalized according to a defined benchmark protocol.

Downstream applications should therefore evaluate Mitra using metrics appropriate to the target scale and operational consequences of error.

Recommended measures include MAE, RMSE, MSE, R², and normalized or task-specific error metrics where appropriate.

## Intersectional Results

Traditional demographic intersectional analysis is **not reported as a general property of the Mitra foundation model**.

Mitra is a domain-general tabular model and its pretraining corpus consists of synthetic learning problems rather than a fixed population of human subjects.

Where Mitra is used in human-centred applications, error should be separately evaluated across relevant demographic, geographic, socioeconomic, institutional, temporal, or operational groups and their intersections.

Such subgroup results cannot be inferred from aggregate foundation-model benchmark performance.

---

# Caveats and Recommendations

## Details

### There Is No Universal Regression Accuracy Percentage

Regression models do not have a single generally meaningful equivalent of classification accuracy. Error values depend on the target's units, scale, distribution, and evaluation protocol.

A registry field labelled "Accuracy (%)" is therefore not a natural performance representation for this model. If such a field is mandatory, its value should be interpreted only according to the registry's explicitly defined regression convention and should not be presented as an intrinsic property of Mitra Regressor.

### Downstream Evaluation Remains Necessary

Before operational use, Mitra should be evaluated on a test dataset representative of the intended deployment setting.

An appropriate evaluation dataset should be independent of fine-tuning data, representative of the deployment population, contain realistic target ranges and difficult cases, include realistic missingness and measurement noise, reflect relevant temporal and operational conditions, permit subgroup analysis where applicable, and avoid target or temporal leakage.

### Compare Against Strong Baselines

Tabular regression problems can often be competitive with comparatively simple baselines. Downstream evaluation should include relevant conventional models and domain-specific naive baselines rather than assuming that a foundation model will be superior by default.

This is particularly important for time-dependent, sparse, or intermittent targets where a persistence, rolling-average, median, or predict-zero baseline may be strong.

### Target Distribution Matters

Highly skewed, heavy-tailed, sparse, zero-inflated, or intermittent targets may materially affect model performance and metric interpretation.

Residual distributions and error across target ranges should be inspected rather than relying only on one aggregate metric.

### Intended Operating Regime

Mitra is primarily designed for **small tabular datasets** and is reported to be particularly effective below approximately **5,000 samples and 100 features**.

Its supported upper limits are approximately **10,000 training samples and 500 features**.

The classification checkpoint's 10-class limit does **not** apply to Mitra Regressor.

### Large-Feature Regression Limitation

The upstream documentation notes that Mitra does not consistently outperform TabPFNv2 on regression tasks with large feature counts.

Model selection should therefore be empirical rather than based solely on aggregate benchmark rank.

### Distribution Shift

Performance may deteriorate when deployment data differ materially from the data used for downstream training or evaluation. Temporal, geographic, demographic, institutional, measurement, process, and target-distribution shifts should be evaluated where relevant.

### Fairness and Subgroup Performance

Published aggregate benchmark results do not establish demographic fairness. Applications involving people should separately evaluate model error across relevant groups and intersections of groups.

### High-Impact Applications

Mitra is a general-purpose tabular foundation model. It has not been established as specifically validated for autonomous decisions in medicine, criminal justice, employment, lending, insurance, public benefits, critical infrastructure, or other high-impact contexts.

Such applications require domain-specific validation, governance, risk assessment, and appropriate human oversight.

### Model Weights Require Architecture Metadata

The uploaded `model.safetensors` artifact contains the learned parameters, while `config.json` defines the architecture required to interpret those parameters.

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

If a registry cannot store `config.json` separately, these values should remain preserved in the model-version documentation.

### Reproducibility

AutoGluon 1.5.0's Mitra loader resolves the model from its Hugging Face repository identifier but does not expose a revision parameter during normal loading. Strict reproduction should verify both checkpoint files against the pinned revision and SHA-256 values recorded in this card.

AutoGluon 1.5.0 also does not fully enable Mitra's global `set_seed` behaviour. A fixed seed may make some stochastic components reproducible but should not be assumed to provide complete bit-for-bit deterministic fine-tuning.

### Recommended Interpretation

Published evidence supports describing Mitra Regressor as a **high-performing general-purpose tabular foundation model within its evaluated small-data regression regime**.

The evidence does not support describing it as having one universal percentage accuracy, uniformly outperforming alternatives on every regression dataset, being robust to arbitrary distribution shift, or being validated by default for high-impact decision-making.

Selection of Mitra for a particular application should be based on evaluation against relevant alternatives using representative downstream data.
