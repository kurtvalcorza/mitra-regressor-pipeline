# Release verification

`tutorials/mitra_regressor_colab.ipynb` (`E2E`) and `tutorials/mitra_regressor_predictor_inference_colab.ipynb`
(`ARTIFACT-INFERENCE`) are **release candidates** until the exact notebook revisions have executed top-to-bottom in a
clean supported runtime. Unit tests, JSON validation, code-cell compilation, and `tools/validate_release_assets.py` are
necessary checks but are **not** runtime evidence under DIMER Notebook Specification 2.2. This file is the durable
release-gate record for both notebooks.

## Automatic coverage (static, every pull request)

CI runs `tools/validate_release_assets.py`, which checks, for each of the two notebooks:

- notebook JSON parses; every code cell compiles as plain Python (no `%`/`!` magics); no persisted outputs or
  execution counts; no unresolved placeholder markers; every code cell is preceded by an explanatory markdown cell;
- exactly the two tutorial notebooks, each named in `tutorials/README.md` with its profile, the notebook-spec version and
  the standalone carrier; `metadata.dimer` declares that profile, spec `2.2`, `standalone: true` and `generated_from`
  (repository, module commit, module path `mitra_pipeline/tutorial_api.py`, module SHA-256, generator);
- the standalone carrier (ST1–ST6, PAR1–PAR3): no clone, repository install or repository import on the primary path
  (the previous pair's `git clone` of this repository is gone); one cell tagged `embedded_module` equal to
  `mitra_pipeline/tutorial_api.py` after the generator's documented rewrite (the `DEFAULT_WEIGHTS_DIR` line); the inline
  `MANIFEST` equal to the committed `weights/mitra-regressor/dimer-base-manifest.json` and the inline `PINS` equal to the
  `pyproject.toml` runtime pins; the notebook byte-identical to `tools/build_notebook.py` output for its template; the
  isolated-environment bootstrap cell (generator /2.2: hash-locked `uv` environment, nothing installed into the kernel, no restart); `NOTEBOOK_SOURCE` recorded in exports;
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
| | | | Standalone ARTIFACT-INFERENCE with an external bundle | | pending — queued to the GPU lane (superseded by the 2026-10-08 rows below) |
| 2026-10-08 | `08667a7` / `8a2ea1c75750` | GitHub Actions `Notebook release execution` run 37747908149 (ubuntu-24.04, runner Python; notebook in the isolated uv environment) | Standalone ARTIFACT-INFERENCE, own-bundle path: the E2E bundle produced in the same job, by `ARTIFACT_ZIP_PATH` + `EXPECTED_ZIP_SHA256`, scoring the 89-row E2E test partition (overlap with support rows asserted 0) | 5m52s job | **PASSED** — CI execution of the path journey (not a Colab run; artifacts kept 30 days by Actions) |
| 2026-10-08 | `08667a7` / `8a2ea1c75750` | Colab CLI 0.7.4 sequential execution, fresh Colab Tesla T4 | Standalone ARTIFACT-INFERENCE default path: pinned `sample-bundle-v1` asset downloaded and SHA-256-verified before extraction, 89 sample rows (no field edited) | 152.3 s | **PASSED** — one pass, no restart, 0 errors: 9/9 code cells in order (`exec.log`); evidence in `docs/execution-evidence/2026-10-08/mitra_regressor_predictor_inference_colab/` |
| 2026-10-08 | `fc56cdd` / `256708eef85f` | Colab CLI 0.7.4 sequential execution, fresh Colab Tesla T4 | Standalone E2E default sample path (no field edited; optional journeys not exercised) | 178.1 s | **PASSED** — one pass, no restart, 0 errors: 10/10 code cells in order (`exec.log`), isolated environment built in 82 s; evidence in `docs/execution-evidence/2026-10-08/mitra_regressor_colab/` |

### Colab CLI execution of the E2E notebook at `fc56cdd` (blob `256708eef85f`) — 2026-10-08

- **Executor:** Colab CLI 0.7.4 sequential execution on a fresh Colab Tesla T4 VM (`colab exec -f`): every code cell in order in one
  kernel, order taken from `exec.log` ("Executing cell k/10"). Not a browser **Run all**; no execution counts; forms not rendered.
- **Notebook:** `tutorials/mitra_regressor_colab.ipynb`, commit `fc56cdd7be65537265b60e956c188c87baaf756e`, blob
  `256708eef85f92109df3118fca94c885bcbdca8f` (fetched byte-exact at the commit; the executed copy's code cells equal the source).
- **Path:** default sample path only, no field edited. Section 1 built the isolated uv environment (74 locked packages, Python
  3.12.12; kernel Python 3.13.15) in 82 s; the model ran on `cuda`.
- **Outcome:** **one pass, no restart, 0 errors**; 10/10 code cells; the carried-module cell has no output by design. Wall 178.1 s.
- **Printed results:** partitions 265 / 88 / 89; Mitra conditioned on 212 of the 265 support rows (53 kept by AutoGluon for its
  internal `Validation score`, printed `-48.1091` = MAE 48.11 on those rows). Holdout MAE / RMSE / R²: Mitra 40.13 / 50.50 / 0.558,
  Random Forest 42.28 / 53.97 / 0.495, LightGBM 44.94 / 57.55 / 0.426, Dummy mean 67.52 / 76.16 / -0.006; test MAE: Mitra 44.50,
  Random Forest 44.67, LightGBM 47.77; printed MAE gap best tree minus Mitra 2.15 (holdout) and 0.17 (test). Evaluation verdict
  `sample-sanity`; `selection_basis` `default:pretrained`; fresh reload **PASS** (predictions within `rtol=1e-6, atol=1e-8`).
- **Exported bundle:** `outputs/mitra_regressor_predictor.zip`, 280,010,193 bytes, SHA-256
  `1b0892040f59a8182e63bc4545b74594d5be72f6b79d97e8f5f9bdbd07f29c48` (printed by the run; the downloaded copy has the same digest).
  It is the source of the `sample-bundle-v1` release asset used by the predictor-inference notebook.
- **Evidence files** (`docs/execution-evidence/2026-10-08/mitra_regressor_colab/`, byte-exact, covered by the `-text` rule):
  executed notebook `3f7a640020b2aed435949007e6c5513eef8195a4af024d1540aed32a8669d5bd`, `run_summary.json` `e252e60b766727e0ba4db28c2aedc0f096aff2a8d77fd56b07902d4a59531dfb`, `exec.log` `43a346373ac5d43f481c933caf8a7723fcc4cb9dc0513bc8167f24f1a3ca316b`, `mitra_regressor_result.json` `a4e9e81286113f6f80dbf822c13b3f7c0c47523f0b34921fb3d4f956a7c91c0d`.
- **Not exercised:** browser Run all, BYOD (single CSV and pre-split), `RUN_FINE_TUNING` on GPU, new-data inference upload,
  the `SEED` activity. The worked answers were checked against this run (NOTEBOOK_SPEC REL13): they hold; LightGBM's test MAE
  (47.77) is a little above the "low-to-mid 40s" range the Section 6 answer gives (rounding-level).

### Colab CLI execution of the ARTIFACT-INFERENCE notebook at `08667a7` (blob `8a2ea1c75750`) — 2026-10-08

- **Executor:** Colab CLI 0.7.4 sequential execution on a fresh Colab Tesla T4 VM (`colab exec -f`), every code cell in order in one
  kernel (order from `exec.log`, "Executing cell k/9"). Not a browser **Run all**; no execution counts; forms not rendered.
- **Notebook:** `tutorials/mitra_regressor_predictor_inference_colab.ipynb`, commit `08667a77edaea726fc257f7e4c60d7411dd64329`, blob
  `8a2ea1c757505b63bce433fda44002cd5f5f3ef6` (fetched byte-exact at the commit; the executed copy's code cells equal the source).
- **Path:** default, no field edited. Section 1 built the isolated uv environment (74 locked packages, Python 3.12.12) in 80 s.
  Section 4 downloaded the pinned release asset `sample-bundle-v1/mitra_regressor_predictor.zip` and matched its whole-archive SHA-256
  `1b0892040f59a8182e63bc4545b74594d5be72f6b79d97e8f5f9bdbd07f29c48` before extraction (`digest_verified: True`; format
  `dimer-autogluon-predictor` version 1). Section 6 scored the 89-row sample input (the producer's independent test partition,
  `overlap_with_support_rows: 0`).
- **Outcome:** **one pass, no restart, 0 errors**; 9/9 code cells; the carried-module cell has no output by design. Wall 152.3 s.
  Evaluation verdict `not-measurable`, `sample_kind` `sample`; four outputs written. Activity (Section 8): `bmi` +0.05 moved the mean
  prediction by +24.46, 88 rows up, 0 down.
- **Evidence files** (`docs/execution-evidence/2026-10-08/mitra_regressor_predictor_inference_colab/`, byte-exact, covered by the
  `-text` rule): executed notebook `65d1899e9ef348f26972089ac225efc36e2dfb66ad5d6a2259a04b0eafd3f22a`, `run_summary.json`
  `29ef8c3c08c887c4d0f00c5562b7d9dca336ab4f1e50e3d1b5696ac4f7ebfa23`, `exec.log` `bdfd37432715114e72f600386753e9ce868584b5b2dc79f95301c00bd829779b`.
- **Earlier run:** the Colab CLI run of `9bb4d3b` (blob `f0ff1a37d6b0`, 9/9, 191.9 s, PASS) is superseded: its Section 4 prose
  named the wrong artifact format, corrected in `08667a7` (prose only).
- **Not exercised:** browser Run all, the own-bundle path on Colab (`ARTIFACT_ZIP_PATH`, covered by the CI run above), the
  `upload` dialogs, the activity with another value. Worked answers checked against this run (REL13): they hold.

## Current status

**Generated tutorial pair: Candidate.** The E2E notebook's current blob `256708eef85f` has one recorded execution, the 2026-10-08 Colab CLI T4 run above (default path only); browser Run all, BYOD, the fine-tuning gate and the activity remain unexercised, and promotion is an integrator's decision. The 2026-09-14 Kaggle T4 row above is evidence for the previous E2E blob (`d32d6f6b45c6`) only: an nbclient execution of the default sample path with a `google.colab` shim, not a browser Colab `Run all`. On 2026-10-08 both notebooks were regenerated (generator /2.2 isolated uv environment, DIMER Notebook Specification 2.2, Notebook Review Framework v1 fixes MRC-M1..M2 / MRC-m1..m6), so under NOTEBOOK_SPEC REL14 they return to Candidate until a run of the exact new blobs is recorded; the earlier row stays as history. The ARTIFACT-INFERENCE companion's current blob `8a2ea1c75750` has a recorded Colab CLI T4 run of its default path (the pinned sample bundle) and a CI run of the own-bundle path, both 2026-10-08. Static validation (`tools/validate_release_assets.py`), nbformat validation, a `compile()` sweep over every code
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


## Maintainer-supplied Colab execution — 2026-09-26

The maintainer reported that this notebook passed an end-to-end Colab run and authorized merging its open PR. The supplied [executed notebook](execution-evidence/2026-09-26/DIMER_FreshRetailNet_MultiModel_Regression_Workshop_v2.ipynb) is preserved byte-for-byte as evidence.

- Reviewed source commit: `087b2fbfbbded007c0fcf3902f167b33b7232da9`.
- Executed-file SHA-256: `ae31481fa1fa5d8d478591ab5a16b9397e7ddf7ebea23447de832220927950af`.
- Independently inspected: 36 executed code cells; zero saved error outputs; terminal completion and exports present.
- Configuration/source comparison: Default controls; cell sources match the reviewed PR exactly.
- Evidence boundary: saved outputs were inspected; execution was not independently repeated. This submission establishes the recorded path, not optional FULL/BYOD paths. Fresh-runtime/restart details beyond the maintainer's explicit prior confirmations are not inferred.

This record supersedes the pending rerun item for the source/configuration above. It does not promote the whole pipeline or close untested optional-path qualification.


## FreshRetailNet regression v2: Notebook Review Framework v1 findings — revision 3.1.1 (2026-09-27)

A review under the Notebook Review Framework v1 (reviewed commit `dd7c8ec`, notebook blob `026741c5`) concluded **Needs revision** for the full interactive/BYOD experience. No default-path blocker was established. It credited the maintainer-supplied Colab execution recorded above. Revision 3.1.1 of `tutorials/DIMER_FreshRetailNet_MultiModel_Regression_Workshop_v2.ipynb` addresses all five findings. `tests/test_workshop_v2_review_fixes.py` executes the notebook's own cell code against synthetic inputs. All of its 11 checks fail on the reviewed revision and pass on 3.1.1. These are logic checks, not model runs.

| Finding | Correction in 3.1.1 | Acceptance check |
|---|---|---|
| **REG-01** (major): a foundation-only frozen selection raised `StopIteration` in the bootstrap | §8.0 fixes and records the paired-comparison reference at the freeze (`comparator.json`: the frozen classical baseline with the best validation score, or none). §8.3 uses that record and never chooses from test scores. With no frozen baseline, it reports each model's own interval and no paired comparison. The comparator is named in every row and exported | A mixed selection records LightGBM; a foundation-only selection records none and still reports every model's interval |
| **REG-02** (major): R² bootstrap draws with one repeated target produced non-finite intervals | R² is `NaN` on a degenerate draw instead of dividing by zero. When any draw is degenerate, the R² intervals are marked `unavailable` with the count, and the point scores are kept. A constant test target is refused with a pointer to MAE or RMSE. §8.0 warns in advance when R² is selected with fewer than 30 test rows | A two-row target `[0, 1]` gives the point R² of 0.96 with every interval `unavailable`; a constant target raises an actionable error |
| **REG-03** (major): tied BYOD targets collapsed the three-band exercise into a one-class task that scored 1.0 | §9.1 shows band coverage per split and scores only when the cut points differ and every band has rows; otherwise it explains and skips. The companion notebook's reference scores are labelled pinned-sample-only, with a BYOD notice | 70/20/10 tied targets are skipped rather than scored; ordinary targets still score a one-band guess at 1/3 |
| **REG-04** (minor): the answers claimed invariance, and the conclusion was MAE-only | The worked answers are labelled as one reference configuration. The conclusion template follows the primary metric chosen in §0.2 and names the comparator | Static checks of both cells |
| **REG-05** (minor): `gain_vs_reference` was a bootstrap mean, not the observed difference | The gain is now the full-test score difference in the favourable direction; paired draws give only its interval | The reported gain equals the difference of the displayed point scores for MAE, RMSE and R² |

Code cells changed, so the saved outputs of the earlier run no longer describe the notebook and were cleared. That run remains in `execution-evidence/2026-09-26/`. **Status: Candidate.** The following exact-revision evidence is required:

- a fresh Colab T4 default `Run all` of revision 3.1.1;
- a foundation-only selection;
- R² as the primary metric;
- positive and negative BYOD runs, including tied targets;
- the documented fast path.

The review's learner walkthrough remains open. Its optional training-median baseline suggestion is not a defect and was not added.


### Maintainer-supplied Colab execution of revision 3.1.1 — 2026-09-27

The maintainer supplied an executed copy of revision 3.1.1 and authorized merging. It is preserved byte-for-byte as [evidence](execution-evidence/2026-09-27/DIMER_FreshRetailNet_MultiModel_Regression_Workshop_v2.ipynb).

- **Source:** branch `fix/regression-review-findings` at `c032386`, notebook blob `97e4a9651559`. All 88 cell ids and sources match exactly.
- **Executed-file SHA-256:** `aab96088bd944311c4d7b460872c7f8f14d66ac5c31735ba2bcb8196252af687`.
- **Runtime:** Colab `gpuType` T4; the four foundation models ran on `cuda`. Host environment:
  - Python 3.13.15
  - NumPy 2.1.3
  - pandas 2.2.3
  - matplotlib 3.10.0
  - scikit-learn 1.6.1
  - LightGBM 4.6.0
- **Execution:**
  - 36 of 36 code cells executed in order (counts 1–36), with no error outputs.
  - Primary metric: MAE.
  - The completion summary reports 9 validation models, 12 frozen test models, no foundation failures, and "Run-all complete".
- **Validation MAE:**
  - TabICLv2 0.36674
  - TabPFN-3 0.37118
  - LightGBM 0.37341
  - Random Forest 0.37644
  - TabDPT 0.39218
  - Mitra 0.41717
- **Test MAE after the freeze:**
  - TabPFN-3 0.42416
  - TabICLv2 0.43286
  - Random Forest 0.43706
  - Mitra 0.44025
  - LightGBM 0.46011
  - TabDPT 0.52836
- **REG-01:** §8.0 fixed the paired comparator (`lightgbm`, the validation-best frozen classical baseline) before the test, and §8.3 used it.
- **REG-05:** each `gain_vs_reference` equals the observed MAE difference. For example, TabPFN-3 is +0.03595 = 0.46011 − 0.42416, with interval [0.01689, 0.05689].
- **REG-02:** all intervals are `available`.
- **REG-03:** §9.1 shows populated bands in every split (cut points 0.5 and 0.9) and scores them.
- **Export:** ZIP SHA-256 `9d4d1d54ad228425…`.
- **Evidence boundary:** the saved outputs were inspected, but the execution was not repeated independently. This record covers only the default path: MAE as the primary metric, all models selected, the pinned sample, and the in-context mode.

This record satisfies the fresh default `Run all` item above. These items remain open:

- a foundation-only selection;
- R² as the primary metric;
- the BYOD runs, including tied targets;
- the fast path.

**Status: Candidate.**


## FreshRetailNet regression v1 (compact edition): Notebook Review Framework v1 findings — revision 3.2.0 (2026-10-03)

A separate review pass under the Notebook Review Framework v1 (reviewed commit `a9a6f05`, notebook blob `6508437e`) concluded **Needs revision**: 0 Blocker, 5 Major, 7 Minor, 1 Suggestion. The review and its probes are in [`reviews/2026-10-03-notebook-review/`](reviews/2026-10-03-notebook-review/), with a verification addendum. Every Major and Minor finding was re-verified against the source before it was fixed; none was refuted. Revision 3.2.0 of `tutorials/DIMER_FreshRetailNet_MultiModel_Regression_Workshop.ipynb` addresses all twelve. `tests/test_workshop_v1_review_fixes.py` executes the notebook's own cell code against synthetic inputs. 22 of its 23 checks fail on the reviewed revision and all pass on 3.2.0 (the 23rd checks that the MAE plot stays MAE). These are logic checks, not model runs.

| Finding | Correction in 3.2.0 |
|---|---|
| **FRR1-M1** (major): after the default Run all, the documented way back to experimenting looped on the frozen error | The 0.3 text, the 8.0 text, a new How-to-use section and the troubleshooting table say exactly what to rerun after starting a new experiment (Section 1.1 onward, **Runtime → Run after**). The prose now explains that Run all freezes by default |
| **FRR1-M2** (major): a foundation-only selection raised `StopIteration` in 8.3 | Port of the guided edition's REG-01 fix: the comparator is fixed and recorded at the freeze, and a selection without a classical baseline keeps each model's own interval |
| **FRR1-M3** (major): R² bootstrap intervals became NaN on small test sets | Port of REG-02: R² intervals are marked unavailable with a count, a constant test target is refused, and the freeze warns in advance |
| **FRR1-M4** (major): the fine-tuning question could not be answered on the default path | Research question 2, the objective and checkpoint question 4 are marked as an optional GPU extension, with the switches, the time limits, the evidence of a weight update and how to answer when it was not run |
| **FRR1-M5** (major): the tutorials README did not say which notebook to use | `tutorials/README.md` has a "Which notebook should I use?" table and states which fixes and differences each edition carries |
| **FRR1-m1**: `gain_vs_reference` was a bootstrap mean | Port of REG-05: the observed full-test difference |
| **FRR1-m2**: plots and the conclusion were MAE-only; ablation sentences mixed partitions | 5.6 and 8.2 plot the primary metric; the template follows it and uses validation scores for both ablations |
| **FRR1-m3**: Run all with **Freeze now** off stopped at 8.1 | 8.1–8.4 skip with a notice and 10.1 exports validation-stage records only, so the exploratory summary is reached |
| **FRR1-m4**: fixed expected-output text did not match the run | The zero-target rate names its split, the archive path line is gone, and 7.1 bands stockout hours as 0, up to 5 and more than 5 hours |
| **FRR1-m5**: identity drift | Title "DIMER Notebook: …", a labelled time estimate, the 2.1 declaration explained, and the shared `_v2` workspace name explained |
| **FRR1-m6**: guided-layer gaps | How to use, Input → Model → Output, infrastructure labels and troubleshooting added; sample answers and the predict–change activity are waived in the opening with a pointer to the guided edition |
| **FRR1-m7**: 4.1 metric code did not produce the tables | 4.1 is labelled an illustration, the unused helper is removed, and the cell checks its values against the pipeline's metric function |

FRR1-S1 (a reference-run note) was not addressed. Code cells changed, so no earlier output describes this revision. **Status: Candidate.** The following exact-revision evidence is required:

- a fresh Colab T4 default `Run all` of revision 3.2.0;
- a run with **Freeze now** off, and the documented new-experiment recovery after a default run;
- a foundation-only selection, and R² as the primary metric;
- positive and negative BYOD runs;
- optionally, one fine-tuning condition on a GPU.

## FreshRetailNet regression workshops (v1 and v2): 2026-10-03 uv isolated environment

Both editions moved together to the uv isolated environment: compact v1 revision 3.2.0 → 3.3.0 (notebook blob `c47cc75d` → `7f216583`) and guided v2 revision 3.1.1 → 3.2.0 (blob `97e4a965` → `5ce1e3e1`). Nothing is installed into the notebook kernel and Run all needs no restart. Section 0.1 checks for a Linux x86_64 kernel with Colab's scikit-learn and LightGBM instead of pip-installing LightGBM. Section 5.2 downloads uv 0.12.15 by SHA-256, creates each model environment with `uv venv --managed-python --python 3.12.12`, and installs the carried `tutorials/requirements-workshop-<stack>.lock.txt` with `--require-hashes --only-binary :all:`. TabDPT's `antlr4-python3-runtime` 4.9.3 has no wheel, so its hash-pinned source archive is built with the locked setuptools. Top-level pins, data, seeds, models and metrics are unchanged. The notebooks are now **Linux x86_64 only** (Colab, Kaggle, Linux Jupyter). `tests/test_workshop_uv_environment.py` covers the change: 12 of its 18 checks fail on the previous revisions, and the other 6 check the lock files only.

A hosted re-run of both revisions passed on 2026-10-03; see the record below. **Status: Candidate.**

### Colab CLI execution of compact v1 revision 3.3.0 and guided v2 revision 3.2.0 — 2026-10-03

Both editions were run separately at commit `aca2655147fc0be2d7b73ae09c9ccdbac196f8ee`, each on its own fresh Colab session. The executed notebooks are preserved byte for byte:

| Edition | Notebook blob | Executed file | SHA-256 | Cells | Session wall |
|---|---|---|---|---|---|
| Compact v1, revision 3.3.0 | `7f21658365e072dc8a96be18ad3fefdd3fbe6b0d` | [`…Workshop_aca2655_colab-cli-t4.ipynb`](execution-evidence/2026-10-03/DIMER_FreshRetailNet_MultiModel_Regression_Workshop_aca2655_colab-cli-t4.ipynb) | `ade39452ae0fb48be34cdcbedc555d32e0b30bac2c523fb2f011e2d5eed20aad` | 33/33 | 701.8 s |
| Guided v2, revision 3.2.0 | `5ce1e3e13554f79e921dab05957abd73822b0012` | [`…Workshop_v2_aca2655_colab-cli-t4.ipynb`](execution-evidence/2026-10-03/DIMER_FreshRetailNet_MultiModel_Regression_Workshop_v2_aca2655_colab-cli-t4.ipynb) | `68323096abafd2fa9ee808f85f8e4bb0172957169315561f3a70b7390d3e4262` | 36/36 | 651.9 s |

- **Source:** each notebook was downloaded from GitHub at the PR head and its Git blob verified before the session.
- **Executor:** Google Colab CLI 0.7.4 on a fresh Colab Tesla T4 session per notebook, through the workspace `colab-cli-serial-test-suite` (`colab new --gpu T4`, `colab exec -f`, `colab stop`). Code cells ran in order in one kernel; this is not a browser Run all, and the CLI records no execution counts, so order is evidenced by its `Executing cell k/N` log.
- **Path exercised:** default controls only — pinned FreshRetailNet sample, MAE as the primary metric, all four foundation models in context, **Freeze now** and the test stage on. The optional activities, fine-tuning, BYOD, the foundation-only selection, R² as the primary metric and the **Freeze now** off path were not run.
- **Outcome:** PASSED, 33/33 (v1) and 36/36 (v2) code cells with no error outputs. Both completion summaries report 9 validation models, 12 frozen test models, no foundation failures, and "Run-all complete".
- **Kernel:** Python 3.13.15 on Linux x86_64; section 0.1 installed nothing. The kernel's classical stack was NumPy 2.1.3, pandas 2.2.3, matplotlib 3.10.0, scikit-learn 1.6.1 and LightGBM 4.6.0.
- **Isolated environments (section 5.2):** in both runs all four uv environments (`mitra`, `tabdpt`, `tabicl`, `tabpfn`) were created with CPython 3.12.12, installed from the hash-locked files, and passed `uv pip check` and the adapter import check. TabDPT's `antlr4-python3-runtime` 4.9.3 was built from its source archive in under 1 s. All four foundation models ran on `cuda`.
- **Setup time:** uv reported about 200 s (v1) and 193 s (v2) preparing and installing packages across the four environments: Mitra 72 s / 77 s, TabDPT 66 s / 61 s, TabICL 60 s / 52 s, TabPFN under 1 s (its packages were already cached). The CLI records no per-cell times, so the wall time of section 5.2 as a whole was not measured.
- **Time against the notebooks' estimates:** the compact edition states about 20–40 minutes of compute on a fresh T4. Its measured session wall, including session start and stop, was 701.8 s (11.7 minutes); the guided edition's was 651.9 s (10.9 minutes). Validation-stage `runtime_seconds` for the foundation models were 10.7–119.5 s (v1) and 10.6–95.8 s (v2); Mitra was the slowest in both. The estimates were not edited.
- **Validation MAE (both editions, identical):** TabICLv2 0.36674, TabPFN-3 0.37118, LightGBM 0.37341, Random Forest 0.37644, TabDPT 0.39218, Mitra 0.41717.
- **Test MAE after the freeze (both editions, identical):** TabPFN-3 0.42416, TabICLv2 0.43286, Random Forest 0.43706, Mitra 0.44025, LightGBM 0.46011, TabDPT 0.52836. The paired-comparison reference fixed at the freeze was `lightgbm`. TabPFN-3's `gain_vs_reference` is +0.03595 with interval [0.01689, 0.05689]. All intervals are `available`.
- **v2 §9.1 demand bands:** cut points 0.5 and 0.9, every band populated in every split, scored as in the earlier run.
- **Report ZIP SHA-256:** v1 `967bbecf5167db2d…`, v2 `3da05f7e25652e7d…`. These are new experiment directories, so the digests differ from earlier runs by construction.
- **Comparison with the most recent recorded hosted run:** the guided edition's last recorded run is revision 3.1.1 above (maintainer-supplied, 2026-09-27). Every printed metric of revision 3.2.0 equals it: the baseline and validation tables, the ablations, the 7.1 error bands, the frozen-test table, the bootstrap intervals and gains, the 8.4 prediction preview and the 9.1 band scores. Only the descriptive `runtime_seconds` column differs, for example TabICLv2 9.24 s → 10.59 s and TabPFN-3 12.04 s → 14.78 s. The compact edition has no earlier recorded hosted run. Its validation, test and bootstrap values equal the guided edition's in both runs; its 7.1 stockout bands differ by design (0, up to 5 and more than 5 hours).
- **Evidence boundary:** the saved outputs were inspected. The journeys not exercised — the interactive activities, the **Freeze now** off path and new-experiment recovery, a foundation-only selection, R² as the primary metric, BYOD, the fast path and fine-tuning — remain open.

This record satisfies the fresh default-path run of both revisions. The other items listed for revisions 3.1.1 and 3.2.0 above remain open. **Status: Candidate.**
