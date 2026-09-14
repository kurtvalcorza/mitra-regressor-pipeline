"""Per-repository template for tools/build_notebook.py (NOTEBOOK_SPEC 1.1 §3.6 standalone carrier) — E2E.

Only the task-specific prose and stage cells live here. Runtime install, the embedded package module
(``mitra_pipeline/tutorial_api.py``, a root-level package: ``package_dir``), and the model pin/stage/verify cell
are produced by the generator from repository sources so they cannot drift from the package. The
ARTIFACT-INFERENCE companion has its own template, ``tools/notebook_template_artifact_inference.py``.
"""
# ruff: noqa: E501  -- markdown prose and code-cell text are kept on single lines for readable rendering

REPO = "mitra-regressor-pipeline"
BADGES = [
    (
        "GitHub",
        "https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white",
        f"https://github.com/kurtvalcorza/{REPO}",
    ),
    (
        "Open In Colab",
        "https://colab.research.google.com/assets/colab-badge.svg",
        f"https://colab.research.google.com/github/kurtvalcorza/{REPO}/blob/main/tutorials/mitra_regressor_colab.ipynb",
    ),
    (
        "Hugging Face",
        "https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-autogluon%2Fmitra--regressor-ffcc4d?style=flat",
        "https://huggingface.co/autogluon/mitra-regressor",
    ),
    (
        "Upstream",
        "https://img.shields.io/badge/Upstream-autogluon%2Fautogluon-181717?style=flat&logo=github&logoColor=white",
        "https://github.com/autogluon/autogluon",
    ),
    ("arXiv", "https://img.shields.io/badge/arXiv-2508.02927-b31b1b.svg", "https://arxiv.org/abs/2508.02927"),
]

TEMPLATE = {
    "package": "mitra_pipeline",
    "package_dir": "mitra_pipeline",  # root-level package (no src/)
    "repo_name": REPO,
    "stem": "mitra_regressor",
    "notebook_name": "mitra_regressor_colab.ipynb",
    "profile": "E2E",
    "mode": "GUIDED",
    "run_all": (
        "Selecting **Run all** in a fresh supported runtime installs the pinned dependencies, stages and digest-verifies the pinned Mitra snapshot, loads scikit-learn's bundled diabetes table (no download), validates the tables into an input manifest and checks the target and split overlap, fits the pretrained Mitra predictor by **in-context conditioning on the training split** (the adaptation stage that runs by default — no gradient update) alongside executable baselines, evaluates MAE/RMSE/R² on the held-out split and writes the evaluation report, exports the deployable predictor bundle and reloads it from disk to prove the fresh boundary. Gradient fine-tuning of the Mitra weights is an optional experiment (`RUN_FINE_TUNING`, off by default, Section 6) because it needs a GPU-sized time budget; a reviewer reading NOTEBOOK_SPEC 2.0 RUN7/FT2 as requiring gradient adaptation on the default path should treat that as an open decision. No repository clone, DIMER worker or service, credential, upload dialog or configuration edit is required (§5)."
    ),
    "byod": (
        "After the sample workflow completes, set `USE_BYOD = True` in Section 4 and re-run from that cell to upload one labelled CSV (or pre-split `train.csv`/`val.csv`/`test.csv`); it enters the same validation, split, in-context fitting, baseline, evaluation, export and fresh-reload cells as the sample (DAT14), and `RUN_NEW_DATA_INFERENCE` in Section 8 scores your own unlabelled rows with the fitted predictor. Expected schema, ceilings and privacy guidance are stated in the Prerequisites and in Section 4; uploads stay inside this runtime. BYOD is optional and never part of the default path."
    ),
    "pipeline_class": "MitraRegressionPipeline",
    "weights_key": "mitra-regressor",
    "modules": ["tutorial_api.py"],
    "entry_module": "tutorial_api.py",
    # `from_pretrained` stages + verifies the snapshot, then stages the verified bytes as the immutable offline
    # Hugging Face snapshot AutoGluon resolves (HF_HUB_OFFLINE). No predictor is built until `fit`.
    "model_load": "MitraRegressionPipeline.from_pretrained(weights_dir=WEIGHTS_DIR)",
    "runtime_imports": ["torch", "numpy", "pandas", "sklearn"],
    "title": "Mitra Regressor — DIMER E2E tabular regression tutorial (standalone)",
    "badges": BADGES,
    "capability": "end-to-end tabular regression with the pinned `autogluon/mitra-regressor` checkpoint through AutoGluon: verified model acquisition, validated support data, in-context evaluation against executable baselines, optional GPU fine-tuning, new-data inference, a deployable predictor bundle and its fresh-boundary reload",
    "intro": (
        "Mitra is an in-context tabular foundation model served through AutoGluon's `TabularPredictor`: with "
        "`fine_tune=False`, `fit` registers the support rows and the model configuration and no weight is "
        "gradient-updated; fine-tuning is an opt-in gate that needs a GPU. The upstream project supplies the model and "
        "the checkpoint; the carried package adds the pinned snapshot scheme, the table validation, capping and "
        "overlap checks, the metric set, the `validate_inputs` / `training_mean_baseline` / `evaluation_report` "
        "helpers, and the archive-safety and artifact-manifest functions. The default sample is scikit-learn's "
        "bundled diabetes table; its metrics are tutorial sanity evidence, not a benchmark or production claim."
    ),
    "learning_objectives": (
        "install the pinned runtime, read what the carried package guarantees, resolve and digest-verify the immutable "
        "upstream checkpoint and stage it as an offline Hugging Face snapshot, load a public sample or your own CSV(s) "
        "and validate them into an input manifest, evaluate the pretrained model on a holdout and an independent test "
        "partition against the training-mean, median, LightGBM and Random Forest baselines, optionally fine-tune on a "
        "GPU with holdout-based selection, write an evaluation report, optionally score new rows, export the "
        "deployable AutoGluon predictor bundle with its manifest and provenance, and prove it reloads from a fresh "
        "directory."
    ),
    "exclusions": (
        "classification, forecasting, calibrated per-prediction uncertainty intervals, or any deployment tolerance "
        "band. Predictions are **continuous point estimates only**; the fine-tuning path runs only on a GPU and only "
        "when `RUN_FINE_TUNING` is switched on."
    ),
    "prerequisites": [
        "- **Runtime:** a fresh supported runtime (Google Colab or Jupyter, Python 3.10–3.13 as required by AutoGluon 1.5.0). The default path runs on CPU and uses CUDA automatically when available; the fine-tuning gate requires a GPU. The pinned `autogluon.tabular[mitra]==1.5.0` install (with its torch) is the largest download of the run.",
        "- **Knowledge:** basic pandas; what a holdout, an independent test partition, MAE, RMSE and R² are.",
        "- **Data:** the default sample is scikit-learn's bundled diabetes table (442 rows, 10 numeric features), loaded from the installed package, so nothing is downloaded and no private data is needed. BYOD (one labelled CSV, or pre-split `train.csv`/`val.csv`/`test.csv` — the repository's `examples/sample-data/` archives can be supplied this way) is selected through `DATA_SOURCE` and is off by default so the sample path runs top-to-bottom without interaction. Do not upload confidential or restricted data to a hosted notebook environment unless you are authorized to do so. Uploaded inputs remain in the notebook runtime; this pipeline does not send them to a third-party inference API.",
    ],
    "cells": [
        {
            "md": (
                "## 4. Load the sample or your own data\n\n"
                "`Sample: Diabetes` (default) is the bundled numeric sanity check, split 60/20/20 into support, holdout and "
                "an independent test partition. `Upload CSV` takes one labelled CSV and makes a seeded random holdout "
                "(it assumes approximately IID rows); `Upload pre-split train/val/test` takes your own partitions "
                "without re-splitting — the repository's `examples/sample-data/*.zip` archives (FreshRetailNet, Insurance "
                "Charges, Ames Housing) ship exactly those three files. Columns listed in `DROP_COLUMNS` are removed "
                "before validation. Rows whose target is missing are **dropped and counted**, exact cross-partition "
                "overlaps are counted, and support rows above `MAX_TRAIN_ROWS` are capped by seeded sampling and "
                "reported. The SHA-256 of the loaded data is printed so the exported provenance can be tied to it."
            ),
            "code": (
                "import hashlib\n\n"
                "from sklearn.datasets import load_diabetes\n"
                "from sklearn.model_selection import train_test_split\n\n"
                "DATA_SOURCE = 'Sample: Diabetes'  # @param [\"Sample: Diabetes\", \"Upload CSV\", \"Upload pre-split train/val/test\"]\n"
                "USE_BYOD = False  # @param {{type:\"boolean\"}}\n"
                "TARGET_COLUMN = 'target'  # @param {{type:\"string\"}}\n"
                "DROP_COLUMNS = ''  # @param {{type:\"string\"}}\n"
                "VALIDATION_SPLIT = 0.20  # @param {{type:\"number\"}}\n"
                "SEED = 42  # @param {{type:\"integer\"}}\n"
                "if USE_BYOD and DATA_SOURCE.startswith('Sample'):\n"
                "    raise ValueError('USE_BYOD=True requires an Upload DATA_SOURCE.')\n"
                "if DATA_SOURCE.startswith('Upload') and not USE_BYOD:\n"
                "    raise ValueError('Set USE_BYOD=True to use an upload DATA_SOURCE.')\n"
                "drop_columns = [c.strip() for c in DROP_COLUMNS.split(',') if c.strip() and c.strip() != TARGET_COLUMN]\n\n"
                "test_data = None\n"
                "if DATA_SOURCE == 'Sample: Diabetes':\n"
                "    dataset = load_diabetes(as_frame=True)\n"
                "    frame = dataset.frame.rename(columns={{dataset.target.name: TARGET_COLUMN}})\n"
                "    train_data, remainder = train_test_split(frame, test_size=0.4, random_state=SEED)\n"
                "    holdout_data, test_data = train_test_split(remainder, test_size=0.5, random_state=SEED)\n"
                "    payloads = {{'sample.csv': frame.to_csv(index=False).encode('utf-8')}}\n"
                "    data_name, sample_kind = 'sklearn-diabetes', 'sample'\n"
                "elif DATA_SOURCE == 'Upload pre-split train/val/test':\n"
                "    from google.colab import files\n"
                "    uploaded = files.upload()\n"
                "    by_base = {{Path(name).name.lower(): payload for name, payload in uploaded.items()}}\n"
                "    missing = sorted({{'train.csv', 'val.csv', 'test.csv'}} - set(by_base))\n"
                "    if missing:\n"
                "        raise RuntimeError(f'Upload train.csv, val.csv, and test.csv together. Missing: {{missing}}')\n"
                "    payloads = {{name: by_base[name] for name in ('train.csv', 'val.csv', 'test.csv')}}\n"
                "    train_data = read_csv_bytes(payloads['train.csv'], 'train.csv')\n"
                "    holdout_data = read_csv_bytes(payloads['val.csv'], 'val.csv')\n"
                "    test_data = read_csv_bytes(payloads['test.csv'], 'test.csv')\n"
                "    data_name, sample_kind = 'pre-split upload', 'BYOD'\n"
                "else:\n"
                "    from google.colab import files\n"
                "    uploaded = files.upload()\n"
                "    csvs = [(name, payload) for name, payload in uploaded.items() if name.lower().endswith('.csv')]\n"
                "    if len(csvs) != 1:\n"
                "        raise RuntimeError('Upload exactly one labelled CSV.')\n"
                "    data_name, payload = csvs[0]\n"
                "    payloads = {{data_name: payload}}\n"
                "    data = read_csv_bytes(payload, data_name)\n"
                "    if not 0.05 <= VALIDATION_SPLIT <= 0.40:\n"
                "        raise ValueError('VALIDATION_SPLIT must be between 0.05 and 0.40.')\n"
                "    train_data, holdout_data = train_test_split(data, test_size=VALIDATION_SPLIT, random_state=SEED, shuffle=True)\n"
                "    sample_kind = 'BYOD'\n"
                "    print('Upload CSV uses a seeded random holdout and assumes approximately IID rows.')\n"
                "DATA_DIGEST = hashlib.sha256(json.dumps({{name: hashlib.sha256(payload).hexdigest() for name, payload in sorted(payloads.items())}}, sort_keys=True).encode()).hexdigest()\n"
                "print({{'sample_kind': sample_kind, 'name': data_name, 'target': TARGET_COLUMN, 'drop_columns': drop_columns, 'train_rows': len(train_data), 'holdout_rows': len(holdout_data), 'test_rows': 0 if test_data is None else len(test_data), 'data_sha256': DATA_DIGEST}})"
            ),
        },
        {
            "md": (
                "## 5. Validate the tables → input manifest, then cap and check overlaps\n\n"
                "`validate_inputs` is the package's public validation stage: it applies exactly the checks "
                "`validate_labeled_frame` applies — unique column names, the target present and numeric, rows with a "
                "missing target dropped and counted, no infinite target, at least `MIN_TRAIN_ROWS` support rows (2 for "
                "a holdout), at most `MAX_FEATURES` features, a support target that varies — and returns an **input "
                "manifest** naming the schema, the observed structure (categorical columns, missing values, the "
                "validation report with dropped and exact-duplicate counts, a target summary) and the verdict. It is "
                "written to `outputs/{stem}_input_manifest.json`. To show what rejection looks like, the cell also "
                "validates a probe with too few rows and records the package's own error message as a finding. The "
                "ceilings are printed before any model runs.\n\n"
                "The holdout and test partitions are then re-ordered to the support schema, exact cross-partition "
                "overlaps are reported (`split_overlap_report`), support rows above `MAX_TRAIN_ROWS` are capped "
                "(`cap_training_rows`), and the trivial training-mean baseline is computed with `training_mean_baseline`. "
                "Everything in Section 6 should be read against that baseline."
            ),
            "code": (
                "os.makedirs('outputs', exist_ok=True)\n"
                "print({{'ceilings': {{'MIN_TRAIN_ROWS': MIN_TRAIN_ROWS, 'MAX_TRAIN_ROWS': MAX_TRAIN_ROWS, 'MAX_FEATURES': MAX_FEATURES}}}})\n"
                "input_manifest = validate_inputs(train_data, target_column=TARGET_COLUMN, drop_columns=drop_columns, names=[data_name + ':train'])\n"
                "input_manifest['inputs'].extend(validate_inputs(holdout_data, target_column=TARGET_COLUMN, drop_columns=drop_columns, min_rows=2, require_variation=False, names=[data_name + ':holdout'])['inputs'])\n"
                "if test_data is not None:\n"
                "    input_manifest['inputs'].extend(validate_inputs(test_data, target_column=TARGET_COLUMN, drop_columns=drop_columns, min_rows=2, require_variation=False, names=[data_name + ':test'])['inputs'])\n"
                "# Demonstrate rejection on a probe that breaks a ceiling; the finding is recorded, not swallowed.\n"
                "try:\n"
                "    validate_inputs(train_data.head(MIN_TRAIN_ROWS - 1), target_column=TARGET_COLUMN, drop_columns=drop_columns)\n"
                "except ValueError as exc:\n"
                "    input_manifest['findings'].append({{'input': 'too-few-rows-probe', 'verdict': 'rejected', 'message': str(exc)}})\n"
                "with open('outputs/{stem}_input_manifest.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(input_manifest, handle, indent=2, ensure_ascii=False)\n"
                "print(json.dumps(input_manifest['inputs'][0], indent=2))\n"
                "print('findings:', input_manifest['findings'])\n\n"
                "train_data, FEATURE_COLUMNS, train_report = validate_labeled_frame(train_data, TARGET_COLUMN, name='train', drop_columns=drop_columns, min_rows=MIN_TRAIN_ROWS, require_variation=True)\n"
                "holdout_data, val_features, _ = validate_labeled_frame(holdout_data, TARGET_COLUMN, name='holdout', drop_columns=drop_columns)\n"
                "ordered = FEATURE_COLUMNS + [TARGET_COLUMN]\n"
                "if set(val_features) != set(FEATURE_COLUMNS):\n"
                "    raise ValueError('train/holdout feature column names do not match.')\n"
                "holdout_data = holdout_data.reindex(columns=ordered)\n"
                "if test_data is not None:\n"
                "    test_data, test_features, _ = validate_labeled_frame(test_data, TARGET_COLUMN, name='test', drop_columns=drop_columns)\n"
                "    if set(test_features) != set(FEATURE_COLUMNS):\n"
                "        raise ValueError('train/test feature column names do not match.')\n"
                "    test_data = test_data.reindex(columns=ordered)\n"
                "overlaps = split_overlap_report({{'train': train_data[ordered], 'holdout': holdout_data, **({{'test': test_data}} if test_data is not None else {{}})}})\n"
                "if any(overlaps.values()):\n"
                "    print('WARNING exact cross-split overlap detected; investigate leakage before interpreting metrics:', overlaps)\n"
                "train_data, cap_report = cap_training_rows(train_data, TARGET_COLUMN, seed=SEED)\n"
                "baseline = training_mean_baseline(train_data[TARGET_COLUMN], holdout_data[TARGET_COLUMN])\n"
                "print({{'train': len(train_data), 'holdout': len(holdout_data), 'test': 0 if test_data is None else len(test_data), 'features': len(FEATURE_COLUMNS), 'cap': cap_report, 'overlaps': overlaps}})\n"
                "if len(FEATURE_COLUMNS) > 100 or len(train_data) > 5_000:\n"
                "    print('Above the <=100-feature / <=5,000-row regime where Mitra is reported to be particularly strong.')\n"
                "print('training-mean baseline on the holdout', baseline)"
            ),
        },
        {
            "md": (
                "## 6. Evaluate pretrained Mitra and executable baselines, then optionally fine-tune\n\n"
                "`pipe.fit(...)` with `fine_tune=False` registers the support rows and the model configuration through "
                "AutoGluon (`fit_mitra_predictor`); no weight is gradient-updated. `regression_metrics` scores the "
                "holdout and, when present, the independent test partition (MAE and RMSE in target units, R² relative "
                "to a constant-mean reference); AutoGluon's own evaluation (`pipe.evaluate`) is printed alongside. "
                "Constant mean/median predictors, LightGBM and Random Forest are fitted on the exact same support rows "
                "(with train-fitted median imputation and ordinal encoding that is never refitted on holdout/test) and "
                "scored on the exact same partitions (EVAL15).\n\n"
                "**Fine-tuning gate (off by default; GPU only).** With `RUN_FINE_TUNING=True` a second predictor is fitted "
                "with `fine_tune=True` for `FINE_TUNE_STEPS`, and it replaces the pretrained predictor **only** if it "
                "beats it on the holdout under `EVAL_METRIC` and the holdout has at least `MIN_SELECTION_HOLDOUT_ROWS` "
                "rows. The independent test is evidence only; a worse test result is surfaced as a warning and never "
                "changes the selection. **Reproducibility boundary.** `SEED` drives the split, the cap and Mitra's "
                "`random_state`; bitwise-identical results across devices and library builds are not promised."
            ),
            "code": (
                "import gc\n"
                "import shutil\n\n"
                "from lightgbm import LGBMRegressor\n"
                "from sklearn.dummy import DummyRegressor\n"
                "from sklearn.ensemble import RandomForestRegressor\n"
                "from sklearn.impute import SimpleImputer\n"
                "from sklearn.preprocessing import OrdinalEncoder\n\n"
                "EVAL_METRIC = 'mean_absolute_error'  # @param [\"mean_absolute_error\", \"root_mean_squared_error\"]\n"
                "BASELINE_TIME_LIMIT = 300  # @param {{type:\"integer\"}}\n"
                "RUN_FINE_TUNING = False  # @param {{type:\"boolean\"}}\n"
                "FINE_TUNE_STEPS = 50  # @param {{type:\"integer\"}}\n"
                "FINE_TUNE_TIME_LIMIT = 600  # @param {{type:\"integer\"}}\n"
                "MAX_MEMORY_USAGE_RATIO = 1.10  # @param {{type:\"number\"}}\n"
                "MIN_SELECTION_HOLDOUT_ROWS = 50\n"
                "PRETRAINED_PATH, FINETUNED_PATH = Path('outputs') / 'mitra-pretrained', Path('outputs') / 'mitra-finetuned'\n"
                "for path in (PRETRAINED_PATH, FINETUNED_PATH):\n"
                "    shutil.rmtree(path, ignore_errors=True)\n"
                "gc.collect()\n"
                "if torch.cuda.is_available():\n"
                "    torch.cuda.empty_cache()\n\n"
                "def score(model, frame):\n"
                "    return regression_metrics(frame[TARGET_COLUMN].to_numpy(dtype=float), model.predict(frame))\n\n"
                "pipe.fit(train_data, target_column=TARGET_COLUMN, eval_metric=EVAL_METRIC, path=PRETRAINED_PATH, fine_tune=False, time_limit=BASELINE_TIME_LIMIT, seed=SEED, max_memory_usage_ratio=MAX_MEMORY_USAGE_RATIO)\n"
                "pretrained_metrics = score(pipe, holdout_data)\n"
                "pretrained_test_metrics = score(pipe, test_data) if test_data is not None else None\n"
                "print('pretrained holdout', pretrained_metrics, '| AutoGluon evaluate:', pipe.evaluate(holdout_data))\n"
                "if pretrained_test_metrics:\n"
                "    print('pretrained independent test', pretrained_test_metrics)\n\n"
                "candidate = candidate_metrics = candidate_test_metrics = None\n"
                "if RUN_FINE_TUNING:\n"
                "    if not torch.cuda.is_available():\n"
                "        raise RuntimeError('Fine-tuning requires a GPU. Choose Runtime -> Change runtime type -> GPU.')\n"
                "    candidate = MitraRegressionPipeline(weights_path=pipe.model_weight_path, config_path=pipe.config_path, snapshot_path=pipe.snapshot_path, device=pipe.device)\n"
                "    candidate.fit(train_data, target_column=TARGET_COLUMN, eval_metric=EVAL_METRIC, path=FINETUNED_PATH, fine_tune=True, fine_tune_steps=FINE_TUNE_STEPS, time_limit=FINE_TUNE_TIME_LIMIT, seed=SEED, max_memory_usage_ratio=MAX_MEMORY_USAGE_RATIO)\n"
                "    candidate_metrics = score(candidate, holdout_data)\n"
                "    candidate_test_metrics = score(candidate, test_data) if test_data is not None else None\n"
                "    print('fine-tuned holdout', candidate_metrics)\n\n"
                "ACTIVE_MODEL, ACTIVE_MODE, SELECTION_BASIS = pipe, 'pretrained', 'default:pretrained'\n"
                "metric_key = {{'mean_absolute_error': 'mae', 'root_mean_squared_error': 'rmse'}}[EVAL_METRIC]\n"
                "if candidate is not None:\n"
                "    if len(holdout_data) < MIN_SELECTION_HOLDOUT_ROWS:\n"
                "        SELECTION_BASIS = f'default:pretrained; holdout-too-small:{{len(holdout_data)}}<{{MIN_SELECTION_HOLDOUT_ROWS}}'\n"
                "    else:\n"
                "        SELECTION_BASIS = f'holdout:{{EVAL_METRIC}}'\n"
                "        if candidate_metrics[metric_key] < pretrained_metrics[metric_key]:\n"
                "            ACTIVE_MODEL, ACTIVE_MODE = candidate, 'fine-tuned'\n"
                "    if candidate_test_metrics and pretrained_test_metrics:\n"
                "        degraded = [k for k in ('mae', 'rmse') if candidate_test_metrics[k] > pretrained_test_metrics[k]] + [k for k in ('r2',) if candidate_test_metrics[k] < pretrained_test_metrics[k]]\n"
                "        if degraded:\n"
                "            print('WARNING independent-test metrics worsened after fine-tuning:', degraded, '(evidence only; never used for selection)')\n"
                "active_metrics = candidate_metrics if ACTIVE_MODE == 'fine-tuned' else pretrained_metrics\n"
                "active_test_metrics = candidate_test_metrics if ACTIVE_MODE == 'fine-tuned' else pretrained_test_metrics\n"
                "print({{'recommended_for_export': ACTIVE_MODE, 'selection_basis': SELECTION_BASIS, 'device': ACTIVE_MODEL.device}})\n\n"
                "# Executable baselines on the exact same partitions.\n"
                "y_train = train_data[TARGET_COLUMN].to_numpy(dtype=float)\n"
                "y_holdout = holdout_data[TARGET_COLUMN].to_numpy(dtype=float)\n"
                "y_test = test_data[TARGET_COLUMN].to_numpy(dtype=float) if test_data is not None else None\n"
                "baseline_rows = []\n"
                "for strategy in ('mean', 'median'):\n"
                "    dummy = DummyRegressor(strategy=strategy).fit(np.zeros((len(y_train), 1)), y_train)\n"
                "    baseline_rows.append({{'model': f'Dummy-{{strategy}}', 'split': 'holdout', **regression_metrics(y_holdout, dummy.predict(np.zeros((len(y_holdout), 1))))}})\n"
                "    if y_test is not None:\n"
                "        baseline_rows.append({{'model': f'Dummy-{{strategy}}', 'split': 'test', **regression_metrics(y_test, dummy.predict(np.zeros((len(y_test), 1))))}})\n"
                "cat_cols = [c for c in FEATURE_COLUMNS if not pd.api.types.is_numeric_dtype(train_data[c])]\n"
                "num_cols = [c for c in FEATURE_COLUMNS if pd.api.types.is_numeric_dtype(train_data[c])]\n"
                "num_imputer = SimpleImputer(strategy='median', keep_empty_features=True) if num_cols else None\n"
                "cat_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1) if cat_cols else None\n\n"
                "def tree_features(frame, fit=False):\n"
                "    parts = []\n"
                "    if num_cols:\n"
                "        values = num_imputer.fit_transform(frame[num_cols]) if fit else num_imputer.transform(frame[num_cols])\n"
                "        parts.append(pd.DataFrame(values, columns=num_cols, index=frame.index))\n"
                "    if cat_cols:\n"
                "        values = cat_encoder.fit_transform(frame[cat_cols].astype(str)) if fit else cat_encoder.transform(frame[cat_cols].astype(str))\n"
                "        parts.append(pd.DataFrame(values, columns=cat_cols, index=frame.index))\n"
                "    return pd.concat(parts, axis=1)[FEATURE_COLUMNS]\n\n"
                "X_train_tree = tree_features(train_data, fit=True)\n"
                "for model_name, model in {{'LightGBM': LGBMRegressor(random_state=SEED, n_estimators=100, verbose=-1), 'RandomForest': RandomForestRegressor(random_state=SEED, n_estimators=100)}}.items():\n"
                "    model.fit(X_train_tree, y_train)\n"
                "    baseline_rows.append({{'model': model_name, 'split': 'holdout', **regression_metrics(y_holdout, model.predict(tree_features(holdout_data)))}})\n"
                "    if test_data is not None:\n"
                "        baseline_rows.append({{'model': model_name, 'split': 'test', **regression_metrics(y_test, model.predict(tree_features(test_data)))}})\n"
                "baseline_rows.append({{'model': f'Mitra-{{ACTIVE_MODE}}', 'split': 'holdout', **active_metrics}})\n"
                "if active_test_metrics:\n"
                "    baseline_rows.append({{'model': f'Mitra-{{ACTIVE_MODE}}', 'split': 'test', **active_test_metrics}})\n"
                "metrics_table = pd.DataFrame(baseline_rows)\n"
                "print(metrics_table.to_string(index=False))\n"
                "print('All values above are current-run tutorial metrics. Lower MAE/RMSE is better; higher R2 is better.')"
            ),
        },
        {
            "md": (
                "## 7. Evaluate → evaluation report\n\n"
                "`evaluation_report` is the package's public evaluation stage and always produces a report. Here it "
                "carries the active model's holdout metrics (`mae`, `rmse`, `r2` — the repository's own metric ids), "
                "the independent-test metrics when a test partition exists, and the training-mean baseline, with the "
                "verdict `sample-sanity`: one seeded split with no dispersion estimate, tutorial evidence rather than a "
                "benchmark; the executable-baseline table is attached. Without a labelled holdout the verdict would be "
                "`not-measurable`. The report is written to `outputs/{stem}_evaluation_report.json`."
            ),
            "code": (
                "report = evaluation_report(active_metrics, baseline=baseline, independent_test=active_test_metrics, n_holdout=len(holdout_data), n_test=None if test_data is None else len(test_data), target_column=TARGET_COLUMN, selection=SELECTION_BASIS, sample_kind=sample_kind, estimation='single seeded split (support/holdout/independent test); no dispersion estimate')\n"
                "report['executable_baselines'] = baseline_rows\n"
                "with open('outputs/{stem}_evaluation_report.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(report, handle, indent=2, ensure_ascii=False)\n"
                "print(json.dumps({{key: report[key] for key in ('verdict', 'reason', 'selection', 'n_holdout', 'n_test')}}, indent=2))"
            ),
        },
        {
            "md": (
                "## 8. Optional new-data point prediction\n\n"
                "Off by default so a top-to-bottom run needs no upload dialog. Switch `RUN_NEW_DATA_INFERENCE` on and upload "
                "one CSV with the support feature columns (extra columns are preserved in the output and not passed to "
                "the model), or set `NEW_DATA_PATH` for a non-interactive executor. `validate_inputs(..., "
                "target_column=None, feature_columns=...)` applies exactly the checks `validate_inference_frame` applies "
                "— unique header, every feature present, no pre-existing `prediction` column. Predictions are "
                "**continuous point estimates only**; on the sample path eight held-out rows are scored instead so a "
                "prediction CSV always exists."
            ),
            "code": (
                "RUN_NEW_DATA_INFERENCE = False  # @param {{type:\"boolean\"}}\n"
                "NEW_DATA_PATH = ''  # @param {{type:\"string\"}}\n"
                "new_data_result = None\n"
                "if RUN_NEW_DATA_INFERENCE:\n"
                "    if NEW_DATA_PATH:\n"
                "        csv_name, payload = os.path.basename(NEW_DATA_PATH), Path(NEW_DATA_PATH).read_bytes()\n"
                "    else:\n"
                "        from google.colab import files\n"
                "        new_upload = files.upload()\n"
                "        csvs = [(name, payload) for name, payload in new_upload.items() if name.lower().endswith('.csv')]\n"
                "        if len(csvs) != 1:\n"
                "            raise RuntimeError('Upload exactly one inference CSV.')\n"
                "        csv_name, payload = csvs[0]\n"
                "    new_data = read_csv_bytes(payload, csv_name)\n"
                "    inference_manifest = validate_inputs(new_data, None, feature_columns=FEATURE_COLUMNS, names=[csv_name])\n"
                "    X_new, extra_columns = validate_inference_frame(new_data, FEATURE_COLUMNS)\n"
                "    out = new_data.copy()\n"
                "    out['prediction'] = ACTIVE_MODEL.predict(X_new)\n"
                "    new_data_result = {{'input': csv_name, 'rows': len(out), 'extra_columns': extra_columns, 'input_manifest': inference_manifest}}\n"
                "else:\n"
                "    out = holdout_data[FEATURE_COLUMNS].head(8).copy()\n"
                "    out['prediction'] = ACTIVE_MODEL.predict(out)\n"
                "    print('Inference upload skipped; eight held-out rows scored instead.')\n"
                "out.to_csv('outputs/{stem}_predictions.csv', index=False)\n"
                "print(out.head())"
            ),
        },
        {
            "md": (
                "## 9. Export the deployable predictor bundle, then prove a fresh reload\n\n"
                "The deployable artifact is the selected AutoGluon predictor directory (ART1), not a replacement "
                "`model.safetensors`: for Mitra it contains the registered support rows, the model configuration and "
                "AutoGluon's serialised state, so it inherits the source data's confidentiality, licensing, retention "
                "and disclosure obligations (ART3/ART7). `tutorial_run_metadata.json` records the base-model identity "
                "and digests, the AutoGluon/Python versions, the target and features, the mode and selection basis, the "
                "data digest and the metrics; `write_artifact_manifest` inventories every file with its size and SHA-256 "
                "(ART5/ART6). The directory is zipped and its SHA-256 printed.\n\n"
                "An in-memory predictor is not evidence that serialisation worked. The cell extracts the exact ZIP into a "
                "fresh directory with `safe_extract_archive` (path, symlink, size and compression-ratio checks; never "
                "`extractall`), verifies the manifest and provenance with `validate_artifact_directory` **before** "
                "deserialising, reloads the predictor and checks that its predictions agree with the in-memory model's "
                "on eight held-out rows within `rtol=1e-6, atol=1e-8` (VER1–VER5). The result JSON then records "
                "everything: predictions, metrics, the evaluation report, the input manifest, the data digest, the "
                "bundle identity, the notebook's source, the model identity, revision and licence, and the runtime."
            ),
            "code": (
                "from datetime import datetime, timezone\n\n"
                "from autogluon.tabular import TabularPredictor\n\n"
                "active_path = Path(ACTIVE_MODEL.predictor.path)\n"
                "run_metadata = {{\n"
                "    'artifact_format': ARTIFACT_FORMAT, 'artifact_format_version': ARTIFACT_FORMAT_VERSION,\n"
                "    'base_model': MODEL_ID, 'base_model_revision': MODEL_REVISION, 'weights_sha256': WEIGHTS_SHA256, 'config_sha256': CONFIG_SHA256,\n"
                "    'notebook_source': NOTEBOOK_SOURCE, 'model_source': 'verified local snapshot (Section 3)',\n"
                "    'autogluon_version': importlib.metadata.version('autogluon.tabular'), 'python_version': platform.python_version(), 'torch_version': torch.__version__, 'device': ACTIVE_MODEL.device,\n"
                "    'problem_type': 'regression', 'target_column': TARGET_COLUMN, 'features': FEATURE_COLUMNS,\n"
                "    'mode': ACTIVE_MODE, 'selection_basis': SELECTION_BASIS, 'seed': SEED, 'data_source': DATA_SOURCE, 'data_sha256': DATA_DIGEST,\n"
                "    'train_rows_before_cap': cap_report['before'], 'train_rows_used': len(train_data), 'train_row_cap_applied': cap_report['applied'],\n"
                "    'holdout_rows': len(holdout_data), 'independent_test_rows': None if test_data is None else len(test_data), 'eval_metric': EVAL_METRIC,\n"
                "    'fine_tuning_requested': RUN_FINE_TUNING, 'fine_tune_steps_requested': FINE_TUNE_STEPS if RUN_FINE_TUNING else None, 'fine_tune_time_limit_seconds': FINE_TUNE_TIME_LIMIT if RUN_FINE_TUNING else None, 'max_memory_usage_ratio': MAX_MEMORY_USAGE_RATIO,\n"
                "    'pretrained_holdout_metrics': pretrained_metrics, 'pretrained_test_metrics': pretrained_test_metrics, 'finetuned_holdout_metrics': candidate_metrics, 'finetuned_test_metrics': candidate_test_metrics,\n"
                "    'prediction_uncertainty': 'point predictions only; no calibrated per-prediction interval', 'exported_at_utc': datetime.now(timezone.utc).isoformat(),\n"
                "}}\n"
                "(active_path / 'tutorial_run_metadata.json').write_text(json.dumps(run_metadata, indent=2), encoding='utf-8')\n"
                "manifest_path = write_artifact_manifest(active_path)\n"
                "archive_base = Path('outputs') / '{stem}_predictor'\n"
                "Path(str(archive_base) + '.zip').unlink(missing_ok=True)\n"
                "archive = Path(shutil.make_archive(str(archive_base), 'zip', root_dir=active_path))\n"
                "archive_digest = sha256_file(archive)\n"
                "print({{'predictor_zip': str(archive), 'zip_sha256': archive_digest, 'artifact_manifest': str(manifest_path)}})\n\n"
                "RELOAD_DIR = Path('outputs') / 'artifact-reload'\n"
                "shutil.rmtree(RELOAD_DIR, ignore_errors=True)\n"
                "safe_extract_archive(archive, RELOAD_DIR)\n"
                "verified_manifest, verified_metadata = validate_artifact_directory(RELOAD_DIR)\n"
                "if verified_metadata['autogluon_version'] != importlib.metadata.version('autogluon.tabular'):\n"
                "    raise RuntimeError('Artifact/runtime AutoGluon version mismatch.')\n"
                "reloaded = TabularPredictor.load(str(RELOAD_DIR))\n"
                "smoke_X = holdout_data[FEATURE_COLUMNS].head(8).copy()\n"
                "np.testing.assert_allclose(ACTIVE_MODEL.predict(smoke_X), predict_regression(reloaded, smoke_X, FEATURE_COLUMNS), rtol=1e-6, atol=1e-8)\n"
                "print('PASS: artifact manifest/provenance verified before deserialisation; predictor reloaded from fresh files; predictions equivalent (rtol=1e-6, atol=1e-8).')\n\n"
                "payload = {{\n"
                "    'predictions': out.to_dict(orient='records'),\n"
                "    'new_data': new_data_result,\n"
                "    'metrics': {{'active_mode': ACTIVE_MODE, 'holdout': active_metrics, 'independent_test': active_test_metrics, 'pretrained_holdout': pretrained_metrics, 'finetuned_holdout': candidate_metrics, 'executable_baselines': baseline_rows}},\n"
                "    'training_mean_baseline': baseline,\n"
                "    'evaluation_report': report,\n"
                "    'input_manifest': input_manifest,\n"
                "    'sample': {{'kind': sample_kind, 'name': data_name, 'source': DATA_SOURCE, 'data_sha256': DATA_DIGEST, 'train_rows': len(train_data), 'holdout_rows': len(holdout_data), 'test_rows': 0 if test_data is None else len(test_data)}},\n"
                "    'artifact': {{'zip': archive.name, 'zip_sha256': archive_digest, 'run_metadata': run_metadata}},\n"
                "    'notebook_source': NOTEBOOK_SOURCE,\n"
                "    'repository_revision': NOTEBOOK_SOURCE['repository_revision'],\n"
                "    'model_id': MODEL_ID,\n"
                "    'model_revision': MODEL_REVISION,\n"
                "    'model_license': MODEL_LICENSE,\n"
                "    'runtime': {{'python': platform.python_version(), 'torch': torch.__version__, 'autogluon': importlib.metadata.version('autogluon.tabular'), 'lightgbm': importlib.metadata.version('lightgbm'), 'numpy': numpy.__version__, 'pandas': pandas.__version__, 'sklearn': sklearn.__version__, 'device': ACTIVE_MODEL.device}},\n"
                "}}\n"
                "with open('outputs/{stem}_result.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(payload, handle, indent=2, ensure_ascii=False)\n"
                "print(sorted(os.listdir('outputs')))"
            ),
        },
    ],
    "closing": (
        "## Interpretation and limits\n\n"
        "Predictions are continuous point estimates in the target's units with no uncertainty interval; any tolerance "
        "band must be chosen on the caller's own labelled, domain-representative data. The evaluation report's "
        "`sample-sanity` verdict names what it is: one seeded split of a public sample with no dispersion estimate — "
        "tutorial evidence that must not be generalised. The executable baselines show when the foundation model adds "
        "value on this table; the fine-tuned variant, when requested, is selected on the holdout only. Rows that are not "
        "independent, targets outside the support range, cross-partition overlaps, tables above the ≤100-feature / "
        "≤5,000-row regime and capped support sets all change results in ways these metrics do not measure.\n\n"
        "Successful execution proves that the recorded repository revision's package, carried in this notebook, can "
        "acquire and digest-verify the pinned checkpoint and stage it offline, validate the demonstrated tables, "
        "register regression support rows through AutoGluon, compute sample metrics against trivial and classical "
        "baselines, write the input manifest and the evaluation report, export the deployable predictor bundle with its "
        "manifest and provenance, and reload an equivalent predictor from that bundle alone — without the repository "
        "being reachable. It does **not** establish benchmark superiority, domain generalisation, fairness, robustness, "
        "calibration, production safety, or deployment fitness.\n\n"
        "**Next experiments:** switch `DATA_SOURCE` to `Upload pre-split train/val/test` with one of the repository's "
        "`examples/sample-data` archives (FreshRetailNet, Insurance Charges, Ames Housing) or your own partitions; enable "
        "`RUN_FINE_TUNING` on a GPU runtime and watch the holdout-based selection and the independent-test warning; feed "
        "the exported `outputs/{stem}_predictor.zip` to the companion predictor-inference notebook in a separate session.\n\n"
        "## References\n\n"
        f"- Repository README: https://github.com/kurtvalcorza/{REPO}/blob/main/README.md\n"
        f"- Repository model card: https://github.com/kurtvalcorza/{REPO}/blob/main/MODEL_CARD.md\n"
        f"- Weight provenance: https://github.com/kurtvalcorza/{REPO}/blob/main/docs/WEIGHTS.md\n"
        "- Upstream model: https://huggingface.co/{MODEL_ID}\n"
        "- Upstream library: https://github.com/autogluon/autogluon\n"
        "- Mitra paper: https://arxiv.org/abs/2508.02927"
    ),
}
