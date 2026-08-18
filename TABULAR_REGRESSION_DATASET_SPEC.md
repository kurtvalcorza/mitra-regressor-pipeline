# Mitra Regressor — Dataset and Validator Specification

## Dataset Format: CSV tables

```
dataset.zip
├── train.csv              (required)
├── val.csv                (optional — a holdout is split from train if missing)
└── test.csv               (optional — scored if present)
```

Each CSV is one row per training example. Column names are the header row. One column is
the regression **target** (numeric); every other column, except those named in
`drop_columns`, is a **feature**. Feature columns may be numeric or categorical — Mitra
handles both.

All CSVs in one dataset must share the same columns.

## Target and Features

| Concept | Rule |
|---|---|
| Target column | Named `target` by default; set the `target_column` preprocessing field to use another name |
| Target type | Must be numeric (regression) |
| Feature columns | Every column except the target and any listed in `drop_columns` |
| Dropped columns | Comma-separated list in `drop_columns` — use for row ids and raw date strings |

## Validation Checks

| Check | Required? | Rule |
|---|---|---|
| `no_nested_zip` | YES | The archive must not contain another `.zip` |
| `train_csv_present` | YES | A `train.csv` must exist in the archive |
| `train_csv_parses` | YES | `train.csv` must parse as CSV |
| `target_column_present` | YES | The configured target column must exist |
| `target_is_numeric` | YES | The target column must be numeric |
| `target_has_values` | YES | The target must have at least one non-null value |
| `feature_columns_present` | YES | At least one feature column must remain after removing target and `drop_columns` |
| `minimum_rows` | YES | At least 50 training rows |
| `val_schema_matches_train` | YES if `val.csv` present | `val.csv` columns must equal `train.csv` columns |
| `test_schema_matches_train` | YES if `test.csv` present | `test.csv` columns must equal `train.csv` columns |
| `row_limit_advisory` | WARNING | Flags tables above the 10,000-row ceiling; the fine-tuner samples down |

## Row Ceiling

Mitra accepts at most **10,000 training rows**. This is a property of the model, not of the
hardware. Tables above the ceiling validate successfully and are seed-sampled to 10,000 rows
before fitting. `max_train_rows` may be lowered but not raised above 10,000.

## Missing Validation Split

If `val.csv` is absent, the fine-tuner carves a deterministic holdout from `train.csv` using
`validation_split` (default 0.2) and the configured seed, and reports the split in
`result.json`. A dataset is never trained without a reported holdout unless it has fewer than
20 rows.

## Result JSON (validator output)

```json
{
  "successful": true,
  "message": "Tabular dataset validation succeeded.",
  "datasetSummary": {
    "fileCount": 2,
    "extensions": { ".csv": 2 },
    "sampleFiles": ["train.csv", "val.csv"],
    "source": "zip",
    "archive": "my-dataset.zip"
  },
  "checks": [
    { "name": "train_csv_present", "successful": true, "message": "Found training table at train.csv." },
    { "name": "target_is_numeric", "successful": true, "message": "Target 'target' is numeric (float64)." }
  ],
  "metadata": {
    "targetColumn": "target",
    "featureColumnCount": 12,
    "rowCount": 8000
  }
}
```

## Dataloader (used by the fine-tuner)

```python
import io
import zipfile
from pathlib import Path

import pandas as pd


def read_csv_from_dataset(dataset_dir: Path, stem: str) -> pd.DataFrame | None:
    """Read <stem>.csv from a raw zip or an unzipped directory."""
    zips = sorted(dataset_dir.glob("*.zip"))
    if zips:
        with zipfile.ZipFile(zips[0]) as zf:
            for member in zf.namelist():
                p = Path(member.lstrip("./"))
                if p.suffix.lower() == ".csv" and p.stem.lower() == stem:
                    with zf.open(member) as handle:
                        return pd.read_csv(io.BytesIO(handle.read()))
        return None
    for path in sorted(dataset_dir.rglob("*.csv")):
        if path.stem.lower() == stem:
            return pd.read_csv(path)
    return None
```

## Fine-tuner Model Call

```python
from autogluon.tabular import TabularPredictor

predictor = TabularPredictor(
    label=target_column,
    problem_type="regression",
    eval_metric="mean_absolute_error",
    path=output_dir / "mitra_predictor",
)
# fine_tune=True (GPU) fine-tunes the weights; fine_tune=False runs zero-shot in-context
# inference. The fine-tuner selects the mode from GPU availability at runtime.
predictor.fit(
    train,
    hyperparameters={"MITRA": {"fine_tune": use_gpu}},
    fit_weighted_ensemble=False,
    time_limit=time_limit_seconds,
)

# Confirm Mitra actually trained before reporting a result.
trained = list(predictor.model_names())
assert any("mitra" in m.lower() for m in trained), f"expected Mitra, got {trained}"

predictions = predictor.predict(val.drop(columns=[target_column]))
```

**Fine-tune versus zero-shot.** Fine-tuning Mitra requires a GPU; on CPU its backward pass hits
an unsupported low-precision path. When no GPU is available, the fine-tuner sets
`fine_tune=False` and Mitra runs **zero-shot** — in-context inference with no weight update.
Zero-shot is faster and CPU-safe, at some cost in accuracy. Each run records the effective
`mode` (`fine-tune` or `zero-shot`) and `device` in its `result.json`.

The saved `TabularPredictor` directory is the model artifact in either mode. It reloads with
`TabularPredictor.load(path)` and predicts on new rows with matching columns.

## Worked Example

[`examples/build_freshretailnet_dataset.py`](examples/build_freshretailnet_dataset.py) builds
a complete, valid dataset zip from the
[FreshRetailNet-50K](https://huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-50K) daily
panel — one row per store-product-day, with sales-history features, the stockout signal, and
promo/holiday/weather covariates, targeting daily sales a chosen number of days ahead. It is a
template for turning any `(entity, date, value)` panel into the CSV contract above.
FreshRetailNet-50K is [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

```
python examples/build_freshretailnet_dataset.py --src <train.parquet> --out ./out --horizon 7
```

## Common Dataset Mistakes

1. **Non-numeric target.** A target column of strings fails validation — regression needs a
   numeric target.
2. **Ids or raw dates left in.** Row ids and unparsed date strings add noise; list them in
   `drop_columns`.
3. **Mismatched splits.** `val.csv` or `test.csv` with different columns than `train.csv`
   fails validation.
4. **A zip inside the zip.** Extract and upload the CSVs at the archive root.
5. **Over 10,000 rows expecting all to be used.** The excess is sampled away; curate the most
   informative rows if the cap matters.
