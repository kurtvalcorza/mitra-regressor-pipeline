# Mitra Regressor standalone Colab tutorials

There are two standalone Colab workflows:

| Notebook | Purpose |
|---|---|
| [`mitra_regressor_colab.ipynb`](mitra_regressor_colab.ipynb) | Acquire/verify Mitra, bring data, evaluate, optionally fine-tune, infer, and export `mitra-predictor.zip` |
| [`mitra_regressor_predictor_inference_colab.ipynb`](mitra_regressor_predictor_inference_colab.ipynb) | Reload an exported `mitra-predictor.zip`, validate a new CSV, run regression inference, and download `predictions.csv` |

Both CSV inference workflows reject duplicate raw headers before pandas can rename them. The BYOD training upload uses the same raw-header protection. Quoted column names and UTF-8 files with a byte-order mark are supported.

### Build/evaluate/export

[![Open build/evaluate/export tutorial in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/mitra-regressor-pipeline/blob/main/tutorials/mitra_regressor_colab.ipynb)

`mitra_regressor_colab.ipynb` is a standalone tutorial for the Mitra Regressor checkpoint distributed through the DIMER Model Repository. It does **not** depend on DIMER Workbench, DIMER APIs, or the DIMER validator/fine-tuner workers.

The tutorial covers:

- DIMER ZIP upload or pinned-upstream checkpoint fallback;
- SHA-256 verification of `model.safetensors` and `config.json`;
- an explicit post-staging resolver check that refuses to continue unless Hugging Face resolves the verified offline snapshot;
- a bundled FreshRetailNet regression sample;
- preservation of the sample's provided `train.csv` / `val.csv` / `test.csv` partitions;
- BYOD single-CSV inspection with a seeded random holdout for approximately IID data;
- pre-split upload for temporal, grouped, embargoed, or leakage-sensitive workflows;
- finite numeric-target validation and training-target variation checks;
- pretrained/in-context Mitra evaluation with conventional positive MAE/RMSE reporting (`mean_absolute_error`, `root_mean_squared_error`);
- optional GPU fine-tuning with an explicit requested step count;
- scalar regression inference; and
- export of run metadata plus a reusable AutoGluon predictor ZIP.

When fine-tuning runs, the notebook recommends the predictor for inference/export using the configured `EVAL_METRIC` on the holdout split. The independent test split is kept out of model selection: it is reported only as evaluation evidence, and the notebook emits a warning when fine-tuning produces mixed or degraded independent-test metrics. The recommended predictor is the one packaged into `mitra-predictor.zip`.

### Use an exported predictor

[![Open exported-predictor inference tutorial in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/mitra-regressor-pipeline/blob/main/tutorials/mitra_regressor_predictor_inference_colab.ipynb)

The inference-only notebook starts from `mitra-predictor.zip`. It installs `autogluon.tabular[mitra]==1.5.0`, safely extracts the archive, optionally verifies the SHA-256 printed by the export step, loads it with `TabularPredictor.load(...)`, verifies that `problem_type` is regression, validates a new CSV, runs `predict()`, and downloads `predictions.csv`. Because AutoGluon deserializes Python model objects, load only predictor ZIPs you created yourself or received from a trusted source; safe ZIP extraction is not a trust guarantee for serialized model contents.

It does **not** reacquire `model.safetensors` or `config.json`, call DIMER, train, fine-tune, or call `predict_proba()`.

```text
DIMER model.safetensors OR pinned upstream checkpoint
        ↓
mitra_regressor_colab.ipynb
        ↓
mitra-predictor.zip
        ↓
mitra_regressor_predictor_inference_colab.ipynb
        ↓
predictions.csv
```

## Runtime note

`autogluon.tabular[mitra]==1.5.0` may replace the PyTorch version preinstalled by Google Colab. The notebooks report the installed PyTorch version, CUDA build, and CUDA availability. If PyTorch had already been imported and pip changes it, restart the session.

`MAX_MEMORY_USAGE_RATIO=1.10` has completed an end-to-end Mitra Regressor run on a standard Tesla T4. The same run showed memory pressure and Mitra reduced `max_samples_support` from 8192 → 4096 → 2048, so 1.10 remains a cautious setting rather than a reason to raise the memory ratio further.

That T4 execution predates the reviewer-hardening patch that added holdout-based predictor selection, predictor-ZIP trust/hash checks, row-drop reporting, and additional input guards. Those changes are covered by repository CI/static tests on the latest PR head; a byte-identical latest-head Colab rerun remains optional additional evidence rather than a prerequisite for understanding the earlier model/runtime measurements.

## Bundled sample dataset

The default data source is [`freshretailnet-h7.zip`](../examples/sample-data/freshretailnet-h7.zip), derived from FreshRetailNet-50K and redistributed under **CC BY 4.0**.

Pinned sample revision: `5625a9eeca94b8c72b9ad1ec78d07ecbaa720903`.

| split | rows |
|---|---:|
| `train.csv` | 4,180 |
| `val.csv` | 1,600 |
| `test.csv` | 1,600 |

Each split has 17 features plus a continuous `target`: daily `sale_amount` seven days ahead. The supplied split is a purged per-series chronological split with a 7-row embargo. The sample is for tutorial/smoke-test use, **not benchmarking**.

## BYOD split guidance

The single-CSV path uses a seeded random holdout and assumes rows are approximately IID. For time-dependent, panel, grouped, lagged, rolling-window, or other leakage-sensitive data, prepare leakage-aware partitions externally and use **Upload pre-split train/val/test**.

The tutorial requires a finite numeric target and non-zero target variation in training. A single uploaded training table, or `train.csv` in a pre-split upload, must retain at least 50 labelled rows after missing-target rows are dropped; `val.csv` and `test.csv` require at least 2 labelled rows. Missing-target drops are reported. Highly intermittent targets should be compared against strong naive baselines. Automatic pretrained-vs-fine-tuned artifact selection additionally requires at least 50 holdout rows; with a smaller holdout, metrics are still shown but the pretrained predictor remains the export default. If fine-tuning is not run, export provenance records `selection_basis=default:pretrained` rather than implying a holdout comparison occurred.

## Model context

- Model: `autogluon/mitra-regressor`
- Task: tabular regression; one continuous numeric prediction per row
- AutoGluon: `1.5.0`
- Revision: `5f277aa8f69042d39d6ac3612aed18bb9279bd95`
- Weights SHA-256: `d8e75c62af0bec2fd404b0ad20a442d951d43ca6d331315cfcc0509b54f2c642`
- Config SHA-256: `2bc1ed5047f7c25368245e8ad32540a5fa28940b1ec05d3f1f454a09ff5384c1`
- Architecture: 12 Transformer layers, dimension 512, 4 heads, `dim_output: 1`
- Pretraining: approximately 45 million synthetic tabular datasets; no real-world pretraining data reported
- Supported ceiling: 10,000 training rows and 500 features
- Particularly strong reported regime: roughly ≤5,000 samples and ≤100 features
- Known caveat: Mitra does not consistently outperform TabPFNv2 on large-feature regression tasks

There is no single universal regression score for the foundation model. Evaluate on the downstream dataset with appropriate error metrics and baselines.

## Export provenance

`tutorial_run_metadata.json` records checkpoint identity, runtime versions, row counts/capping, selected metric, requested fine-tuning schedule, memory ratio, and holdout/independent-test metrics. Error metrics are exported in conventional positive form.

## AI use and provenance

These tutorials were developed with substantial AI assistance using **GPT-5.6 Sol High** under human direction and review.

- AI model/configuration: **GPT-5.6 Sol High**
- Provider/client: **OpenAI / ChatGPT**
- Agent Relay role: **Builder**
- Base-model developer: **AutoGluon team, Amazon Web Services (AWS)**
- DIMER role: distributor of the pinned `model.safetensors` artifact, not model developer

AI attribution is **provenance, not sign-off** and does not independently verify correctness.
