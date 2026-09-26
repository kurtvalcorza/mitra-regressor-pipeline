# Release verification

`tutorials/mitra_regressor_colab.ipynb` (`E2E`) and `tutorials/mitra_regressor_predictor_inference_colab.ipynb`
(`ARTIFACT-INFERENCE`) are **release candidates** until the exact notebook revisions have executed top-to-bottom in a
clean supported runtime. Unit tests, JSON validation, code-cell compilation, and `tools/validate_release_assets.py` are
necessary checks but are **not** runtime evidence under DIMER Notebook Specification 1.1. This file is the durable
release-gate record for both notebooks.

## Automatic coverage (static, every pull request)

CI runs `tools/validate_release_assets.py`, which checks, for each of the two notebooks:

- notebook JSON parses; every code cell compiles as plain Python (no `%`/`!` magics); no persisted outputs or
  execution counts; no unresolved placeholder markers; every code cell is preceded by an explanatory markdown cell;
- exactly the two tutorial notebooks, each named in `tutorials/README.md` with its profile, the notebook-spec version and
  the standalone carrier; `metadata.dimer` declares that profile, spec `1.1`, `standalone: true` and `generated_from`
  (repository, module commit, module path `mitra_pipeline/tutorial_api.py`, module SHA-256, generator);
- the standalone carrier (ST1–ST6, PAR1–PAR3): no clone, repository install or repository import on the primary path
  (the previous pair's `git clone` of this repository is gone); one cell tagged `embedded_module` equal to
  `mitra_pipeline/tutorial_api.py` after the generator's documented rewrite (the `DEFAULT_WEIGHTS_DIR` line); the inline
  `MANIFEST` equal to the committed `weights/mitra-regressor/dimer-base-manifest.json` and the inline `PINS` equal to the
  `pyproject.toml` runtime pins; the notebook byte-identical to `tools/build_notebook.py` output for its template; the
  pinned-install cell with its restart-on-stale-import guard; `NOTEBOOK_SOURCE` recorded in exports;
- `MODEL_ID`/`MODEL_REVISION` are bound only in the carried module cell (and repeated in the inline manifest, which the
  notebook asserts against the module before fetching), the revision is a 40-hex immutable commit, and the same identity
  string appears in `README.md`, `MODEL_CARD.md`, and `docs/WEIGHTS.md` with no stray revisions;
- the profile-specific public-API calls — E2E: `stage_missing_files`, `verify_snapshot`,
  `MitraRegressionPipeline.from_pretrained(weights_dir=WEIGHTS_DIR)` (which stages the verified bytes into the offline HF
  cache through `stage_verified_hf_snapshot`), `validate_inputs` (with the too-few-rows rejection probe),
  `validate_labeled_frame`, `split_overlap_report`, `cap_training_rows`, `training_mean_baseline`, `fit` (in-context,
  `fine_tune=False`), the GPU-guarded `fine_tune=True` candidate behind `RUN_FINE_TUNING`, `DummyRegressor` / LightGBM /
  Random Forest on the same partitions, `evaluation_report` with the independent test, inference-mode `validate_inputs` +
  `validate_inference_frame`, `write_artifact_manifest`, `safe_extract_archive` + `validate_artifact_directory` before
  `TabularPredictor.load` on the fresh reload and the `rtol=1e-6, atol=1e-8` equivalence check; companion: the required
  whole-archive digest (`ALLOW_UNVERIFIED_ARTIFACT` gated off), `safe_extract_archive`, `validate_artifact_directory`, the
  base-model identity/digest comparison, the AutoGluon/Python compatibility checks before `TabularPredictor.load`,
  inference-mode `validate_inputs` with a rejection probe, `predict`, a `not-measurable` `evaluation_report` — the ceiling
  prints, the four exports per notebook, the learner-facing statements (point estimates only, dropped rows counted, no
  gradient update, fine-tuning gate, reproducibility boundary, verify-before-deserialise, trust boundary, no artifact created
  in the companion) and the gated-off form parameters (`USE_BYOD`, `RUN_FINE_TUNING`, `RUN_NEW_DATA_INFERENCE`,
  `ALLOW_UNVERIFIED_ARTIFACT`); forbidden patterns (credential-in-URL, own-repository clone/install, a `git+https://`
  dependency without a 40-hex SHA, a mutable `revision='main'`, `worker.run(` / `worker_cli(` / `subprocess.run([` outside
  the install cell, direct `urllib.request` / `TabularPredictor(` / `predictor.fit(` / `hyperparameters=` / `zipfile.ZipFile(`
  / `hf_hub_download(` / `sklearn.metrics` use **outside the carried module cell**, `trust_remote_code=True`, `pickle.load`,
  `torch.load(`, `extractall(`; in the companion also `load_diabetes(`, `shutil.make_archive(`, `.fit(`, `write_artifact_manifest(`);
- `STATUS.md`, `README.md` and `tutorials/README.md` agree on one release-status token and no document makes an
  unsupported release-grade, production-readiness or benchmark claim;
- `MODEL_CARD.md` front matter (license, base model), single H1, placeholder/claim hygiene and the
  `## Checkpoint and Artifact Provenance` section (the card predates the MODEL_CARD_SPEC heading structure; the structural
  checks are switched off in the validator until the fleet's card pass reaches this branch).

CI also runs `ruff`, `tools/build_notebook.py --check` for both templates, `scripts/check_shared.py`,
`scripts/check_contract.py`, `scripts/validate_colab_tutorial.py` (the repository's own spec-1.1 checks), the two
`scripts/test_*.py` regression suites (`test_tutorial_api.py` against the package, `test_colab_csv_headers.py` against the
package and the generated notebooks) and the offline unit suite (`tests/`: `test_role_helpers.py`, `test_notebook_parity.py`,
`test_companion_parity.py`; injected downloader, no weights, AutoGluon never imported). These are source/provenance and unit
checks. They are **not** execution evidence.

## Executor paths

| Path | Runtime | Role |
|---|---|---|
| Google Colab (supported user path) | Colab CPU or GPU runtime, Python 3.10–3.13 | The runtime the tutorials are written for; a clean top-to-bottom run here is promotion evidence |
| Kaggle CLI kernel | Kaggle kernel, Python 3.10–3.13 image | Reproducible clean-room executor of the same class; the notebook is pushed verbatim plus one leading shim cell that provides `google.colab` and chdirs to a scratch directory (no repository checkout is needed — the notebooks are standalone). For the companion the shim also places the E2E run's `outputs/mitra_regressor_predictor.zip`, its printed SHA-256 and a separately generated unlabelled CSV, and sets `ARTIFACT_ZIP_PATH` / `EXPECTED_ZIP_SHA256` / `NEW_DATA_PATH` |
| GitHub Actions `notebook-release.yml` (previous pair) | hosted runner, real kernels | Executed the previous repository-installing pair (`/content` workspace, `DIMER_*` / `MITRA_*` env vars); it must be re-pointed at the standalone pair (`outputs/` workspace, the form parameters above) before it counts again |

## Supported release verification procedure

Before changing the registry status from `Candidate` to `Release-grade`:

1. resolve the exact PR/commit head under review and confirm static CI is green;
2. open the exact E2E notebook revision in a new CPU (or CUDA) runtime with **no repository checkout** and a clean model cache;
3. run it top-to-bottom without editing implementation cells (form parameters at their defaults: `DATA_SOURCE = 'Sample: Diabetes'`,
   `USE_BYOD = False`, `RUN_FINE_TUNING = False`, `EVAL_METRIC = 'mean_absolute_error'`, `RUN_NEW_DATA_INFERENCE = False`);
4. verify that Section 1 reports `NOTEBOOK_SOURCE.repository_revision` equal to the module commit recorded in
   `metadata.dimer.generated_from` and that the installed core package versions equal the inline `PINS` (= `pyproject.toml`:
   `autogluon.tabular[mitra]==1.5.0`, `lightgbm==4.6.0`, `huggingface-hub==0.36.2`; torch and friends are AutoGluon's transitive pins);
5. verify every default-path stage completes:
   - pinned runtime installed from the inline `PINS` with no GitHub access;
   - the carried module cell executes (defines `MitraRegressionPipeline`, the validation/metric helpers and the archive-safety
     functions) with no import of the repository package;
   - pinned `autogluon/mitra-regressor` acquisition at the immutable revision through the package: the inline `MANIFEST` is
     asserted against the module identity and written to `weights/mitra-regressor/`, `stage_missing_files(WEIGHTS_DIR, allow_download=True)`
     reports the two manifest entries (`model.safetensors` 302,683,140 bytes, `config.json` 81 bytes) on a clean runtime,
     `verify_snapshot` returns the manifest dict, `from_pretrained` reports `source == 'local-snapshot'` and the offline HF
     snapshot path;
   - the diabetes sample split 60/20/20 with its data SHA-256 printed; the ceilings (`MIN_TRAIN_ROWS` 50, `MAX_TRAIN_ROWS` 10,000,
     `MAX_FEATURES` 500) surfaced; `validate_inputs` writes `outputs/mitra_regressor_input_manifest.json` (three accepted tables,
     one recorded rejection finding from the too-few-rows probe); overlap and cap reports printed;
   - `fit` (in-context conditioning through AutoGluon), `regression_metrics` on holdout and independent test, `pipe.evaluate`,
     `training_mean_baseline`; the fine-tuning gate skipped; Dummy mean/median, LightGBM and Random Forest on the same rows;
   - `evaluation_report` writes `outputs/mitra_regressor_evaluation_report.json` with verdict `sample-sanity`, the three metric ids,
     the independent-test block and the training-mean baseline;
   - eight held-out rows scored into `outputs/mitra_regressor_predictions.csv` (inference upload skipped);
   - the predictor bundle `outputs/mitra_regressor_predictor.zip` written with `tutorial_run_metadata.json` and `artifact_manifest.json`,
     extracted with `safe_extract_archive` into `outputs/artifact-reload/`, `validate_artifact_directory` passes before
     `TabularPredictor.load`, and the reloaded predictor's predictions equal the in-memory model's within `rtol=1e-6, atol=1e-8`;
     `outputs/mitra_regressor_result.json` written with `NOTEBOOK_SOURCE`, model revision, model licence, runtime versions and device;
6. in a **second** clean runtime, run the exact companion notebook revision with `ARTIFACT_ZIP_PATH` pointing at a copy of the E2E
   run's bundle, `EXPECTED_ZIP_SHA256` set to its printed digest and `NEW_DATA_PATH` at a separately generated unlabelled CSV (or
   supply the files through the upload dialog); verify the digest, archive, manifest, base-model and runtime-compatibility checks
   pass before `TabularPredictor.load`, that `validate_inputs(..., target_column=None, ...)` writes
   `outputs/mitra_regressor_predictor_inference_input_manifest.json` with one recorded rejection finding, and that the
   `not-measurable` evaluation report, `..._predictions.csv` and `..._result.json` are written;
7. verify the exports exist and the interpretation sections match the observed paths;
8. record the notebook Git blob ids, commit, runtime (platform, Python, PyTorch, AutoGluon, device), model identifier and immutable
   revision, whether the model cache was clean, outcome, produced outputs, and any warning or applicable `SHOULD` deviation in the table below;
9. record no access tokens or other secrets.

A known-failing default path in the supported runtime blocks release.

## Recorded executions

Notebook identity is the Git blob id of the notebook file (verify with `git rev-parse <commit>:tutorials/<notebook>`). Wall
times, when recorded, are the sum of per-cell times reported by the executor and include installs and the model download;
they are measurements for the stated runtime, not general estimates.

### Manual clean-runtime evidence

| Date (UTC) | Commit / notebook blob | Executor | Path exercised | Wall | Outcome |
|---|---|---|---|---|---|
| 2026-09-14 | `7a4efc7` / `d32d6f6b45c6` | Kaggle T4 (`kurtvalcorza/dimer-nb2-mitra-regressor` v2) | Standalone E2E default sample path | 159.4 s | **PASSED** — 9/9 ok code cells executed cleanly, 9 files, 605 MB staged |
| | | | Standalone ARTIFACT-INFERENCE with an external bundle | | pending — queued to the GPU lane |

## Current status

No clean-runtime execution of the standalone notebooks has been recorded yet; clean GPU execution evidence for the E2E path is now recorded below. Static validation (`tools/validate_release_assets.py`), nbformat validation, a `compile()` sweep over every code
cell, and the offline unit suite passed on the tutorial source at the candidate revision, which is necessary but not
sufficient. The registry status remains **Candidate** until a reviewer confirms a recorded run against the notebook blobs
under review and an integrator promotes it; promotion is not performed by the builder. Facts a reviewer should weigh:
`stage_missing_files` was exercised only with an injected downloader in the unit suite (the real `hf_hub_download` fetch
into a fresh `weights/mitra-regressor/` has not been executed); `from_pretrained` builds no predictor — AutoGluon loads the
model inside `fit`; the previous workflow executions covered the old repository-installing notebooks, not this carrier; the
default sample changed from the repository-hosted FreshRetailNet archive to scikit-learn's diabetes table (the repository
archives are reachable only through the upload path now); and the standalone carrier itself — executing the carried module
cell in a runtime that has no repository checkout — has been validated statically only (parity PASS, carrier probe with the
package import blocked), never run. The clean runs will be the first execution of the standalone path, of the staging path,
of the new helpers (`validate_inputs`, `training_mean_baseline`, `evaluation_report`) and of `MitraRegressionPipeline`
against the real weights.

## FreshRetailNet guided regression v2: saved execution and remaining gates

This record applies to `tutorials/DIMER_FreshRetailNet_MultiModel_Regression_Workshop_v2.ipynb`, separately from the generated diabetes E2E/companion pair above. Status remains **Candidate**.

| Evidence | Recorded fact | Limit |
|---|---|---|
| Original executed artifact | Commit `4ffea2c7d543a34768027519e006a898412f9324`; notebook Git blob `7405af2e89c6585c3f425eb15b98e888490a25d2` | Identity of the saved run, not the revised prose |
| Saved outputs | 36/36 code cells have execution counts; 0 saved error outputs; terminal run-all/export summary present | Saved output inspection is not an independent fresh-runtime execution |
| User-confirmed execution conditions | Maintainer explicitly confirmed on 2026-09-26 that the original v2 run used a fresh Colab runtime and default Run all, with no manual restarts or rerunning cells | User attestation plus inspected saved outputs; not an independent executor rerun and not execution of the revised prose |
| Runtime declaration | Notebook metadata specifies Colab GPU / T4 | Metadata alone does not establish the actual hardware or clean-start conditions |
| Guided-text revision | Executable cells and their outputs/counts are preserved; new orientation, predictions and activity instructions | This prose revision has not been rerun end to end |

The original commit above now has user-confirmed fresh-runtime/default-path evidence, supported by its inspected saved outputs. For REL1/REL10 review of the revised teaching artifact, record the exact revised notebook commit and blob, the executor/date, Python and package versions, actual device, form settings, clean-runtime/model-cache conditions and exported report digest. Run the revised notebook top to bottom in supported Colab, then separately follow the interactive activity before freezing. Do not promote from saved outputs or static checks alone. The notebook's `clean_runtime_evidence: pending` metadata remains accurate for this unverified revised artifact.

### BYOD verification recipe (REL12 remains pending)

Use a fresh Colab runtime and a new Section 0.3 experiment for each case. Record notebook and ZIP digests, runtime details, settings and outputs.

1. **Positive path:** supply a representative pre-split ZIP containing `train.csv`, `val.csv` and `test.csv`, each with the 17 documented numeric features and numeric `target`, consistent columns, finite targets and at least two rows per split (use enough training/support rows for all selected model conditions). Retain meaningful non-overlapping temporal partitions. Set `USE_BYOD=True`, `BYOD_METHOD="path"`, and `BYOD_ZIP_PATH` to the uploaded/staged ZIP. Keep the remaining default model and evaluation settings. Confirm explicit-path acquisition and schema checks, then run the full downstream baseline/foundation comparison, freeze, saved-artifact reload, independent test, inference preview and report export. Record successful selected-model receipts and the terminal summary, not just CSV acceptance.
2. **Negative path:** make a separate copy with `lag_1` removed from all three CSVs. Use the same path-mode controls in a fresh experiment. Confirm Section 2.1 raises `Expected features are missing: ['lag_1']` before fitting, freezing or exporting a successful model result. Preserve that diagnostic and record the rejected archive digest.
3. Record both outcomes against the exact notebook revision. Local acquisition/schema probes are useful preliminary evidence only; they do not satisfy full downstream BYOD execution or clean Colab release gates.

### Local BYOD validation probe — 2026-09-26

The revised v2 notebook's actual path-mode acquisition, ZIP staging and full schema-validation cells were executed locally with its embedded `CORE_SOURCE` in a temporary directory (Python 3.12, pandas 3.0.5, NumPy 2.5.2). Only the three BYOD form assignments were overridden in memory. A representative ZIP repacked from the repository sample was accepted with 4,180 training, 1,600 validation and 1,600 test rows. A separate ZIP with `lag_1` removed from all splits was rejected with `Expected features are missing: ['lag_1']` before any model execution. These are acquisition/validation-only probes; no foundation models, artifact reload, test evaluation or export were executed, so REL12 remains open.
