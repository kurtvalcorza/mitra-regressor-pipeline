# Release verification

`tutorials/mitra_regressor_colab.ipynb` (`E2E`, kind `TUTORIAL`) is a **release candidate** until the exact notebook revision
has executed top-to-bottom in a clean supported runtime. Unit tests, JSON validation, code-cell compilation, and
`tools/validate_release_assets.py` are necessary checks but are **not** runtime evidence under DIMER Notebook Specification 3.0.
This file is the durable release-gate record for the generated notebook. NOTEBOOK_SPEC 3.0 folded the predictor-inference
companion (`tutorials/mitra_regressor_predictor_inference_colab.ipynb`, deleted) into the E2E notebook's Section 10
external-artifact section; the companion's recorded runs below stay as history and are evidence for that deleted notebook only.

## Automatic coverage (static, every pull request)

CI runs `tools/validate_release_assets.py`, which checks, for the generated notebook:

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
  `TabularPredictor.load` on the fresh reload and the `rtol=1e-6, atol=1e-8` equivalence check; Section 10: the pinned
  `SAMPLE_ARTIFACT_URL` / `SAMPLE_ARTIFACT_SHA256` download (`fetch_pinned_artifact`), the own-bundle digest
  (`ALLOW_UNVERIFIED_ARTIFACT` gated off), `safe_extract_archive`, `validate_artifact_directory`, the base-model
  identity/digest comparison, `check_data_contract` (SART10), the AutoGluon/Python compatibility checks before
  `TabularPredictor.load`, inference-mode `validate_inputs` with a rejection probe, `predict`, a `not-measurable`
  `evaluation_report` — the ceiling prints, the four exports plus the two Section 10 outputs, the learner-facing statements (point estimates only, dropped rows counted, no
  gradient update, fine-tuning gate, reproducibility boundary, verify-before-deserialise, trust boundary, Section 10 never
  reading this run's export) and the gated-off form parameters (`USE_BYOD`, `RUN_FINE_TUNING`, `RUN_NEW_DATA_INFERENCE`,
  `ALLOW_UNVERIFIED_ARTIFACT`, `USE_OWN_ROWS`); forbidden patterns (credential-in-URL, own-repository clone/install, a `git+https://`
  dependency without a 40-hex SHA, a mutable `revision='main'`, `worker.run(` / `worker_cli(` / `subprocess.run([` outside
  the install cell, direct `urllib.request` / `TabularPredictor(` / `predictor.fit(` / `hyperparameters=` / `zipfile.ZipFile(`
  / `hf_hub_download(` / `sklearn.metrics` use **outside the carried module cell**, `trust_remote_code=True`, `pickle.load`,
  `torch.load(`, `extractall(`, and `shutil.make_archive(` (the export is `deterministic_zip`); a `github.com/kurtvalcorza`
  reference anywhere except the exact pinned sample-artifact URL, as a whole token, in the carried module (ST1);
- `STATUS.md`, `README.md` and `tutorials/README.md` agree on one release-status token and no document makes an
  unsupported release-grade, production-readiness or benchmark claim;
- `MODEL_CARD.md` front matter (license, base model), single H1, placeholder/claim hygiene and the
  `## Checkpoint and Artifact Provenance` section (the card predates the MODEL_CARD_SPEC heading structure; the structural
  checks are switched off in the validator until the fleet's card pass reaches this branch).

CI also runs `ruff`, `tools/build_notebook.py --check`, `scripts/check_shared.py`,
`scripts/check_contract.py`, `scripts/validate_colab_tutorial.py` (the repository's own spec-1.1 checks), the two
`scripts/test_*.py` regression suites (`test_tutorial_api.py` against the package, `test_colab_csv_headers.py` against the
package and the generated notebooks) and the offline unit suite (`tests/`: `test_role_helpers.py`, `test_notebook_parity.py`,
`test_external_artifact_section.py`; injected downloader, no weights, AutoGluon never imported). These are source/provenance and unit
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
6. on the same default path, Section 10 prints the pinned URL, SHA-256 and producer before downloading, accepts the asset only at
   `SAMPLE_ARTIFACT_SHA256`, passes the archive, manifest, base-model, data-contract and runtime-compatibility checks before
   `TabularPredictor.load`, refuses the missing-column probe, and writes
   `outputs/external_artifact/mitra_regressor_external_predictions.csv` (89 rows) and
   `outputs/external_artifact/mitra_regressor_external_result.json` (verdict `not-measurable`, artifact `zip_sha256` equal to the pin);
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
| 2026-10-14 | `4c44ff0` / `dfb253c68e55` | Colab CLI 0.7.4 sequential execution, fresh Colab Tesla T4 | Guided v2 (`DIMER_FreshRetailNet_MultiModel_Regression_Workshop_v2.ipynb`, revision 3.2.0 with the REG2 fixes, spec 2.5) default path (no field edited; activities, Freeze now off, BYOD and fine-tuning not exercised) | 593.9 s | **PASSED** — one pass, no restart, 0 errors: 36/36 code cells in order (`exec.log`); evidence in `docs/execution-evidence/2026-10-14/` |

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
  executed notebook `3f7a640020b2aed435949007e6c5513eef8195a4af024d1540aed32a8669d5bd`, `run_summary.json` `e252e60b766727e0ba4db28c2aedc0f096aff2a8d77fd56b07902d4a59531dfb`, `exec.log` `43a346373ac5d43f481c933caf8a7723fcc4cb9dc0513bc8167f24f1a3ca316b` (as first committed; redacted 2026-10-22, now `8a384d833e71bb2a1060c3053b4e737f7995a2c1f5ececa16a522064793462da`), `mitra_regressor_result.json` `a4e9e81286113f6f80dbf822c13b3f7c0c47523f0b34921fb3d4f956a7c91c0d`.
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
  `29ef8c3c08c887c4d0f00c5562b7d9dca336ab4f1e50e3d1b5696ac4f7ebfa23`, `exec.log` `bdfd37432715114e72f600386753e9ce868584b5b2dc79f95301c00bd829779b` (as first committed; redacted 2026-10-22, now `a7555e02ac44410ae62303a3b1fb1685751db79f5aaaac4c9116f2415c9dee1f`).
- **Earlier run:** the Colab CLI run of `9bb4d3b` (blob `f0ff1a37d6b0`, 9/9, 191.9 s, PASS) is superseded: its Section 4 prose
  named the wrong artifact format, corrected in `08667a7` (prose only).
- **Not exercised:** browser Run all, the own-bundle path on Colab (`ARTIFACT_ZIP_PATH`, covered by the CI run above), the
  `upload` dialogs, the activity with another value. Worked answers checked against this run (REL13): they hold.

### Producing run of the pinned sample artifact (2026-10-22)

Colab CLI 0.7.4 `colab exec -f`, fresh Colab Tesla T4, default path (`Sample: Diabetes`) of
`tutorials/mitra_regressor_colab.ipynb` at `0423aaa4224b6874266d8b32c349dcfbb40612bc` (blob `256708eef85f`), exit code 0,
148.1 s wall, input notebook unchanged; the run's own fresh reload verified the manifest and provenance before deserialisation and
matched predictions at `rtol=1e-6, atol=1e-8`. Its exported `outputs/mitra_regressor_predictor.zip` (SHA-256
`850613f21b303a2f2dca49e54e1c0a736a1fd65cfc31d81fad03a79a841e7a19`, 280,010,179 bytes, printed by the run) was published as the
release asset `sample-artifact-diabetes-v1/mitra_regressor_predictor.zip`, which Section 10 pins as `SAMPLE_ARTIFACT_URL` /
`SAMPLE_ARTIFACT_SHA256`. It was written by that revision's `shutil.make_archive`, so its digest is not reproducible by re-running;
later exports use `deterministic_zip`. This run is evidence for blob `256708eef85f` only, not for the NOTEBOOK_SPEC 3.0 revision.
Evidence in `docs/execution-evidence/2026-10-22/mitra_regressor_colab_0423aaa/` (byte-exact, `-text`). The local run directory
was redacted to `<local-run-dir>` before the first commit; SHA-256 captured → committed:

| File | Captured | Committed |
|---|---|---|
| `run_summary.json` | `0a9771cb9014c72c5f96ed9891fff11131377defc2bc7171ccec68fea1e9162d` | same (no local path) |
| `exec.log` | `0eeba7e69ac18669e4e1367c63ddd62357fa206c5755b71038076042bfc3b0cc` | `9b0effa4cd9679c156e9d6c1fbf59182a8e48e99253d63c6a9d55ba1f8fc711b` |
| `mitra_regressor_colab_0423aaa_colab-cli-t4_output.ipynb` | `353a61fdf205f693a35a6a592337ba3b02a6233ab93b2ce3bd80050971181b2d` | same (no local path) |
| `suite.log` | `35b63ba626a5e4d6c1936024e2c25dec572f967a37e71115d63b0e692624c66d` | `14155ff1d42610a9edbd255e98486bed832bb63fe58f5ed93b810a64b4fdda05` |

### Serialization scan of the pinned sample artifact (2026-10-23, retroactive)

ARTIFACT_SPEC 2.0 (ml-worker#90): SER9 (a): AutoGluon has no faithful non-code-executing format for a `TabularPredictor`, so the predictor stays pickle-based (including `models/Mitra/model.pt`, which pickles the whole model object). It needs a recorded trust note (SER6, under "Artifact contract" in [`tutorials/README.md`](../tutorials/README.md)) and a scan with an acceptable outcome (SER13). SER13 asks for the scan **before** publication; this asset was published on
2026-10-22, before 2.0 existed, so this is a **retroactive** scan of the published bytes, not the pre-publication control. The
next sample artifact is to be scanned before upload and published only at the digest that passed.

The release asset `sample-artifact-diabetes-v1/mitra_regressor_predictor.zip` was downloaded on 2026-10-23; its SHA-256 (`850613f21b303a2f2dca49e54e1c0a736a1fd65cfc31d81fad03a79a841e7a19`) matches the pinned value. `scan.py`
(picklescan 1.0.5, joblib 1.6.0, numpy 2.5.4, Python 3.13) extracted it with member-path checks, reached every nested archive,
and judged each pickle file on its own: **dangerous** if any global is dangerous, **incomplete** if no recorded method read the
whole file, otherwise **clean**. Content decides what is a pickle, not the file name, and every archive (PyTorch zips
included) is unpacked so each member gets its own verdict.

| File | Kind | Verdict |
|---|---|---|
| `mitra_regressor_predictor.zip` | archive | extracted |
| `predictor.pkl` | pickle | clean |
| `learner.pkl` | pickle | clean |
| `models/trainer.pkl` | pickle | clean |
| `models/Mitra/model.pt` | pytorch-zip | extracted |
| `models/Mitra/model.pkl` | pickle | clean |
| `utils/data/y_val.pkl` | pickle | clean |
| `utils/data/y.pkl` | pickle | clean |
| `utils/data/X_val.pkl` | pickle | clean |
| `utils/data/X.pkl` | pickle | clean |
| `utils/attr/Mitra/y_pred_proba_val.pkl` | pickle | clean |
| `models/Mitra/model.pt!/model/data.pkl` | pickle | clean |

- **Result:** 10 pickle files; clean=10 dangerous=0 incomplete=0; `scan_status=0`. Dangerous globals: none.
- **Globals (distinct, all files):** 58, by top-level package: autogluon (27), numpy (8), pandas (7), torch (7), builtins (3), collections (2), networkx (2), __builtin__ (1), sklearn (1). The per-file lists with picklescan's
  classification are in the report.
- **Evidence:** `docs/execution-evidence/2026-10-23/ser13-sample-artifact-scan/` holds `scan-report.txt` exactly as `scan.py`
  wrote it (it records only paths inside the asset), `scan.py`, and `test_scan.py`, its regression tests (dangerous globals
  behind joblib buffers or beside joblib files, protocol-0 pickles under other names, pickle/zip polyglots, every PyTorch zip
  member, object-dtype .npy files in every dtype spelling, object records after numeric ones in a multi-record .npy
  (also inside an .npz, and at both resume points after a STOP inside a frame), extension-registry opcodes (also in
  streams that fail to load), compressed prefixes in front of an archive, streams at both continuations of a frame
  (also with different memos), protocol-2 compatibility names, pickles after a record without NumPy magic, non-UTF-8
  member names, torch payloads in complete pickles or under any qualified split of the loader name, object
  arrays and protocol-2 bytes, memo-aliased globals (also across consecutive streams), dangerous streams after a clean
  one or inside one protocol-4 frame, a pickle in front of the asset itself, zip/tar
  polyglots, compressed (including zlib of any window size or with delayed output, and LZMA-alone) and truncated
  pickles, streams that share a memo or sit behind a STOP inside a frame, streams after a legacy checkpoint's five
  pickles and tails that are not its raw storages, nested archives that try to overwrite queued members, duplicate,
  escaping and malformed members, relative member paths such as `home/` in the report, and unreadable assets).
  `scan.py` exits 0 clean, 1 dangerous, 2 incomplete, 3 could not scan.
- **Scope:** the threat model at the top of `scan.py` (frozen 2026-10-24) says what the scan covers. Data a loader
  decodes with no format marker, code that is not pickle, and harm through callables picklescan rates safe or
  suspicious are out of scope; review findings inside the scope are fixed, findings outside it are recorded.

### First hosted run of the migrated notebook (2026-10-22)

Colab CLI 0.7.4 `colab exec -f` on a fresh Colab Tesla T4, default path of `tutorials/mitra_regressor_colab.ipynb` on PR #45 at
`92c627caee1c005ce4dc82546fff4892f50f24cb` (blob `3261520866b3`, Notebook Spec 3.0; the second pass of §32.2), started
2026-10-22T03:18:22Z, 171.4 s wall, exit code 0, input notebook unchanged; executed cell sources equal the blob (40 cells). The
fresh reload of the run's own export passed (predictions equivalent at `rtol=1e-6`; its deterministic export printed
`zip_sha256` `28f8b9f4b95ce36b4ca6e51f021ccc19800a8fbf5262687b3e3c61e506df2639`). **Section 10 ran end to end on the default
path:** it downloaded the pinned release asset, verified SHA-256 `850613f21b303a2f2dca49e54e1c0a736a1fd65cfc31d81fad03a79a841e7a19`
before extraction, rebuilt the Mitra regressor from the bundle (10 features, on CUDA), scored 89 rows and wrote
`outputs/external_artifact/mitra_regressor_external_predictions.csv` and `..._external_result.json` (verdict `not-measurable`, as
designed: the section scores rows as new input without targets). Evidence in
`docs/execution-evidence/2026-10-22/mitra_regressor_colab_92c627c/`; the local run directory was redacted to `<local-run-dir>`
before the first commit; SHA-256 captured → committed:

| File | Captured | Committed |
|---|---|---|
| `run_summary.json` | `db31f2985d1a31b6b25aaf82ffd74751c60f0d00bea569a8f4b61656e39760a5` | same (no local path) |
| `exec.log` | `6aebde180b8ef29c6dc238b694a5a5094a38914dedb4e1355fa67073daaa35e2` | `92166074a7f5b1d3dcb2e823332a5425b560455f9b558ed5d8ca3fa32898dd7d` |
| `mitra_regressor_colab_92c627c_colab-cli-t4_output.ipynb` | `22d30709f4a3f12233e87de7f3095d80c0de98cc78520d706e1b5247b739618e` | same (no local path) |
| `suite.log` | `a9075c968c7652ea327b05c3d651a1cf4bae143b977b5c3ef4fafea1295522a3` | `8b69163cbb3bc6d06c614035bf5a4d8dcd1991e3919c87a3b2f68bfa94445b7e` |

### Optional-activity run (2026-10-22)

`f294525` / committed blob `32615208`, `SEED = 7`. The Section 11 activity is a re-run of Sections 4–7 with `SEED = 7`, so it was
run as a run copy of committed blob `32615208` with one override, `SEED = 42` → `7` (run copy blob `a09930c08cfb`, SHA-256
`95c8d0c339bf146324daedfcb17ffc8bf68196aae96a73724a950b8c9a759a6a`; it differs from the committed blob only in that assignment
literal and is committed beside the evidence as `run_copy.ipynb`). Colab CLI 0.7.5 `colab exec -f` (a scripted run, not a browser
Run all), fresh Colab Tesla T4, at `main` `f294525a0113f7f4f1bc007f4e3c15471cfe026a`, started 2026-10-22T13:33:10Z, 184.0 s wall,
exit code 0, sequential, input (the run copy) unchanged, 13/13 code cells in the executor log, 0 errors, verdict PASS. With the
seed-7 split (265 / 88 / 89 rows) the notebook printed pretrained holdout MAE 45.4541 / RMSE 55.6893 / R² 0.5111 beside the
training-mean baseline's MAE 68.5416, and independent test MAE 41.8734 / R² 0.4749 (the seed-42 default split gives holdout MAE
40.1328 / R² 0.5576 and test MAE 44.4957 / R² 0.4819 in the fine-tune run below: the split alone moves the holdout MAE by about
5); the fresh reload printed PASS and Section 10 verified the pinned asset (`trusted_digest: verified (pinned SAMPLE_ARTIFACT_SHA256)`).
Evidence in `docs/execution-evidence/2026-10-22/mitra_regressor_colab_f294525-activity/`. The local run directory was redacted to
`<local-run-dir>` before the first commit; SHA-256 captured → committed:

| File | Captured | Committed |
|---|---|---|
| `mitra_regressor_colab_f294525_colab-cli-t4_activity.ipynb` | `8a0c966b7f9a12cc8f51606fd7de61e8d102832339a2e1e0d489904fa2a0b104` | same (no local path) |
| `exec.log` | `a949650a30e7c8c6c03cb4f7263abf0d171d2d6479ed8bb867ada81f2d996579` | `812f50271e80c0a182a50cf645c787a8a4b692f6b1c961687bf10f19413ad242` (2 replacements) |
| `run_summary.json` | `7ae8187955997f5e12454ae109d90a2532d01eb6149a577516159a6edec84439` | same (no local path) |
| `run_copy.ipynb` | `95c8d0c339bf146324daedfcb17ffc8bf68196aae96a73724a950b8c9a759a6a` | same (no local path) |

### `RUN_FINE_TUNING` run (2026-10-22)

`f294525` / committed blob `32615208`, `RUN_FINE_TUNING = True`. A run copy of committed blob `32615208` with one override,
`RUN_FINE_TUNING = False` → `True` (run copy blob `57234ac30a85`, SHA-256 `01a6bd208009cf320be5d687241adebc7d22973a2dd32928af39344b4e933a2a`;
it differs from the committed blob only in that assignment literal and is committed beside the evidence as `run_copy.ipynb`).
Colab CLI 0.7.5 `colab exec -f` (a scripted run, not a browser Run all), fresh Colab Tesla T4, at `f294525`, started
2026-10-22T13:54:24Z, 361.6 s wall, exit code 0, sequential, input (the run copy) unchanged, 13/13 code cells in the executor log,
0 errors, verdict PASS. On the default diabetes sample the GPU fine-tune ran on cuda. Holdout: pretrained MAE 40.1328 / RMSE
50.4991 / R² 0.5576, fine-tuned MAE 40.1924 / RMSE 50.5632 / R² 0.5565 (slightly worse). Independent test: pretrained MAE 44.4957 /
RMSE 54.7465 / R² 0.4819, fine-tuned 44.6101 / 54.8549 / 0.4798, and the notebook printed "WARNING independent-test metrics worsened
after fine-tuning: ['mae', 'rmse', 'r2'] (evidence only; never used for selection)". Selection stayed on the holdout
(`'recommended_for_export': 'pretrained', 'selection_basis': 'holdout:mean_absolute_error'`); the fresh reload and Section 10
passed as on the default path. Evidence in `docs/execution-evidence/2026-10-22/mitra_regressor_colab_f294525-finetune/`. The local
run directory was redacted to `<local-run-dir>` before the first commit; SHA-256 captured → committed:

| File | Captured | Committed |
|---|---|---|
| `mitra_regressor_colab_f294525_colab-cli-t4_finetune.ipynb` | `2da199d91deb38e7e97c537a84a02a0983ef0517650c7a8397a609378769c778` | same (no local path) |
| `exec.log` | `3020bc2bddc08e1edb3509ae1da1c764b60eb3fce94c0509faec8fa03766bf9f` | `084e47fb81ac44310825d9e0890002e924bfba0ecb27c55805cf49ea5b0b3b4b` (2 replacements) |
| `run_summary.json` | `63cff95c35817786cd261dd73da1a26c37fdbed632acebb7dff546d0c634fb69` | same (no local path) |
| `run_copy.ipynb` | `01a6bd208009cf320be5d687241adebc7d22973a2dd32928af39344b4e933a2a` | same (no local path) |

### Default-path run of the refusal-display fix (2026-10-23)

`191e631` / committed blob `1269d721`, default path. Colab CLI `colab exec -f` (a scripted run, not a browser Run all), fresh Colab
Tesla T4, PR #49 branch `fix/refusal-display` at `191e6315758195f64ba50aaf4806a72f70345327` (blob
`1269d721599dda405f78c6ec1e4a0136bf791dab`), started 2026-10-23T08:38:52Z, 270.9 s wall, exit code 0, input notebook unchanged,
13/13 code cells in the executor log, 0 errors, verdict PASS. The notebook printed pretrained holdout MAE 40.1328 / RMSE 50.4991 /
R² 0.5576 beside the training-mean baseline's MAE 67.5162, and independent test MAE 44.4957 / RMSE 54.7465 / R² 0.4819 (the same
values as the `f294525` fine-tune run's pretrained model on blob `32615208`); the deterministic export printed `zip_sha256`
`1ce8868f3122e799ccbc5c6f74afff43667b51482b3b540f1e120e7a20286a6c`; the fresh reload printed PASS; Section 10 verified the pinned
asset before extraction (`trusted_digest: verified (pinned SAMPLE_ARTIFACT_SHA256)`, SHA-256
`850613f21b303a2f2dca49e54e1c0a736a1fd65cfc31d81fad03a79a841e7a19`) and rebuilt the Mitra regressor from the bundle (10 features,
on CUDA). Evidence in `docs/execution-evidence/2026-10-22/mitra_regressor_colab_191e631/`. The local run directory was redacted to `<local-run-dir>` before the first commit; SHA-256 of the executed notebook equals `run_summary.json` `executed_sha256`; SHA-256 captured → committed:

| File | Captured | Committed |
|---|---|---|
| `mitra_regressor_colab_191e631_colab-cli-t4.ipynb` | `67d65806a4b9bec0354d4a9712d307fc2353e0e4b54edb71d939d6d25812ce67` | same (no local path) |
| `exec.log` | `62a62dcf6efa013440904cf57d8d549f6101c23f6b774493240fe444eae006a2` | `2b8f3f0324ab6afbd0e41b244cfc80b89333558201dd230d427a7701f83f25dc` (2 replacements) |
| `run_summary.json` | `4a4794830f3667bc81ea90fd6e44b17ad437803dc1729d4d5dba2356e4a10a83` | same (no local path) |

### Optional-activity run of `1269d721` (2026-10-23)

`191e631` / committed blob `1269d721`, `SEED = 7`. A run copy of committed blob `1269d721` with one override, `SEED = 42` → `7`
(run copy blob `3479cdf92bdd`, SHA-256 `99e69a1aa9bfc2ac9a0506f02578aa2de5c0dd96d63e5c75eb973eec76c28781`; it differs from the
committed blob only in that assignment literal and is committed beside the evidence as `run_copy.ipynb`). Colab CLI
`colab exec -f` (a scripted run, not a browser Run all), fresh Colab Tesla T4, at `191e631`, started 2026-10-23T08:39:07Z,
153.4 s wall, exit code 0, input (the run copy) unchanged, 13/13 code cells in the executor log, 0 errors, verdict PASS. With the
seed-7 split the notebook printed pretrained holdout MAE 45.4541 / RMSE 55.6893 / R² 0.5111 beside the training-mean baseline's
MAE 68.5416, and independent test MAE 41.8734 / RMSE 53.8821 / R² 0.4749 (equal to four decimals to the `f294525` activity run on
blob `32615208`); the fresh reload printed PASS and Section 10 verified the pinned asset
(`trusted_digest: verified (pinned SAMPLE_ARTIFACT_SHA256)`). Evidence in `docs/execution-evidence/2026-10-22/mitra_regressor_colab_191e631-activity/`. The local run directory was redacted to `<local-run-dir>` before the first commit; SHA-256 of the executed notebook equals `run_summary.json` `executed_sha256`; SHA-256 captured → committed:

| File | Captured | Committed |
|---|---|---|
| `mitra_regressor_colab_191e631_colab-cli-t4_activity.ipynb` | `0b3814b68b88165671b76c5afb5342ff39cafb42c5e700b28ded0dd6ae008236` | same (no local path) |
| `exec.log` | `f29d16796eb13162a8993cec77a8ff6a2a565c7873a91392323ced250ff52016` | `ac6d6ce901a76877ea1c133daa47ff809e5166221927a79b13de51531d40ec13` (2 replacements) |
| `run_summary.json` | `ffefa4f7658bc2df57818633268d1e3ad47af6603dc33359502bfbcfbe635f03` | same (no local path) |
| `run_copy.ipynb` | `99e69a1aa9bfc2ac9a0506f02578aa2de5c0dd96d63e5c75eb973eec76c28781` | same (no local path) |

### `RUN_FINE_TUNING` run of `1269d721` (2026-10-23)

`191e631` / committed blob `1269d721`, `RUN_FINE_TUNING = True`. A run copy of committed blob `1269d721` with one override,
`RUN_FINE_TUNING = False` → `True` (run copy blob `98cf48cfb4a6`, SHA-256
`37397606ad600cba4d684c161000e1eb8b4ded42986b39fffe36ad2792f5a9c4`; it differs from the committed blob only in that assignment
literal and is committed beside the evidence as `run_copy.ipynb`). Colab CLI `colab exec -f` (a scripted run, not a browser Run
all), fresh Colab Tesla T4, at `191e631`, started 2026-10-23T08:41:52Z, 225.9 s wall, exit code 0, input (the run copy) unchanged,
13/13 code cells in the executor log, 0 errors, verdict PASS. On the default diabetes sample the GPU fine-tune ran on cuda.
Holdout: pretrained MAE 40.1328 / RMSE 50.4991 / R² 0.5576, Mitra-fine-tuned MAE 40.1924 / RMSE 50.5632 / R² 0.5565 (slightly
worse). Independent test: pretrained MAE 44.4957 / RMSE 54.7465 / R² 0.4819, Mitra-fine-tuned 44.6101 / 54.8549 / 0.4798, and the
notebook printed "WARNING independent-test metrics worsened after fine-tuning: ['mae', 'rmse', 'r2'] (evidence only; never used
for selection)". Selection stayed on the holdout and kept the pretrained model
(`'recommended_for_export': 'pretrained', 'selection_basis': 'holdout:mean_absolute_error'`); the fine-tuned predictor was saved
(`mitra-finetuned` appears in the printed `outputs/` listing). Fine-tuning did not improve on the pretrained model in this run.
The fresh reload and Section 10 passed as on the default path. Evidence in `docs/execution-evidence/2026-10-22/mitra_regressor_colab_191e631-finetune/`. The local run directory was redacted to `<local-run-dir>` before the first commit; SHA-256 of the executed notebook equals `run_summary.json` `executed_sha256`; SHA-256 captured → committed:

| File | Captured | Committed |
|---|---|---|
| `mitra_regressor_colab_191e631_colab-cli-t4_finetune.ipynb` | `fdb5335c81892eb6631fb9182d14a5905db00aa0716ef3c1f9411148e61d3f34` | same (no local path) |
| `exec.log` | `2438f2f09dbfbffedf2ed73e0dca71014019ec7009fabf0123314819d437565a` | `ad71726f7d701be1ea85faf82e88f5947924e393682c6a7bd10ee0e324ccbcfc` (2 replacements) |
| `run_summary.json` | `e0686f33c2272daea1663c343f865bf818604288840389edfff492c709cbcc34` | same (no local path) |
| `run_copy.ipynb` | `37397606ad600cba4d684c161000e1eb8b4ded42986b39fffe36ad2792f5a9c4` | same (no local path) |

## Current status

The 2026-10-23 refusal-display fix (branch `fix/refusal-display`, NOTEBOOK_SPEC 3.0 DAT19/DAT20: a refused user input is shown as one `Refused: <file name>: <rule>` line with no traceback, and the run stops at that cell) changed the notebook blob from `32615208` to `1269d721`. **Generated E2E tutorial: Candidate (2026-10-23).** The current E2E notebook blob is `1269d721` (`tutorials/mitra_regressor_colab.ipynb` at `191e631` on `fix/refusal-display`, PR #49; Notebook Spec 3.0 with the Section 10 external-artifact section and the pinned release asset `sample-artifact-diabetes-v1`). It has three recorded hosted runs, the 2026-10-23 records above, all Colab CLI `colab exec -f` (scripted, not a browser Run all) on a fresh Colab Tesla T4, exit 0, 13/13 code cells in the executor log, 0 errors, input unchanged, verdict PASS: the default path (270.9 s; the fresh reload passed and Section 10 verified the pinned asset before extraction), the Section 11 activity (`SEED = 7` run copy, 153.4 s) and the `RUN_FINE_TUNING` branch (`RUN_FINE_TUNING = True` run copy, 225.9 s; the fine-tuned holdout MAE 40.1924 was worse than the pretrained 40.1328, so the pretrained model was kept). The owner's 2026-10-22 ruling (the optional activity and the `RUN_FINE_TUNING` branch run once on the current blob before promotion) is satisfied for `1269d721` by the last two. Under REL14 the `92c627c` default-path run and the `f294525` activity and fine-tune runs are history for blob `32615208`, not evidence for `1269d721`. The notebook stays **Candidate** until these two are recorded for this blob:

- **REL12:** a BYOD run in which one representative input is accepted through the downstream stages and one incompatible input is refused;
- **REL2:** a browser run of the interactive journeys (BYOD upload dialog, optional-stage checkboxes, own-artifact ZIP); the scripted CLI runs count for the default path, the activity and the fine-tune branch only.

Promotion is an integrator decision; the builder does not change the status.

**History (the 2.2 pair, superseded; not evidence for `32615208`, REL14).** The previous blob `256708eef85f` has the 2026-10-08 Colab CLI T4 run above (default path only). The 2026-09-14 Kaggle T4 row above is evidence for the previous E2E blob (`d32d6f6b45c6`) only: an nbclient execution of the default sample path with a `google.colab` shim, not a browser Colab `Run all`. On 2026-10-08 both notebooks were regenerated (generator /2.2 isolated uv environment, DIMER Notebook Specification 2.2, Notebook Review Framework v1 fixes MRC-M1..M2 / MRC-m1..m6), so under NOTEBOOK_SPEC REL14 they return to Candidate until a run of the exact new blobs is recorded; the earlier row stays as history. The ARTIFACT-INFERENCE companion's then-current blob `8a2ea1c75750` (deleted with the companion) has a recorded Colab CLI T4 run of its default path (the pinned sample bundle) and a CI run of the own-bundle path, both 2026-10-08. Static validation (`tools/validate_release_assets.py`), nbformat validation, a `compile()` sweep over every code
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

## FreshRetailNet guided regression v2: W14 review findings (REG2) — 2026-10-12

Review of blob `5ce1e3e1` (revision 3.2.0, `main` at `3db6e84`): 0 Blocker, 2 Major, 11 Minor, 2 Suggestion. The fixes
keep the revision label 3.2.0 and record themselves in `metadata.dimer.review_fixes`. No default-path model or
metric computation changed (static assessment; hosted run pending); the additions are records (`comparator.json` and
manifest fields, `validation_selected` and `evaluation_status` columns), a test-exposure log file and wording.

- **REG2-M1 (owner ruling B):** the 8.3 intervals are described as an illustrative row-level (IID) bootstrap that is
  probably too narrow, with a note on the 11 paired comparisons; superiority wording is gone. Follow-up: rebuild the
  dataset with an evaluation-only series id for a cluster bootstrap.
- **REG2-M2:** 8.0 records the validation-selected model (best validation primary metric among the frozen full-feature
  conditions; ties to the alphabetically first key) with the comparator; 8.2/8.3 mark it and 8.3 prints its test score
  as the confirmatory result; the conclusion template reports it.
- **REG2-m1 (owner ruling B):** the fixed-recipe comparison is disclosed (Section 4 and the limitations). Follow-up: a
  validation-tuned, early-stopped LightGBM baseline.
- **REG2-m7 (owner ruling A):** a runtime-level `test_exposure_log.jsonl` under the workspace root, written by 3.1 (test
  EDA) and 8.1 (test scoring), is not reset by Start new experiment; 8.0 copies earlier experiments' entries into
  `comparator.json` and 10.1 into `experiment_manifest.json` (`prior_test_exposure`, `test_previously_exposed`).
- **REG2-m2:** with Freeze now off, 8.1–8.4 skip with a notice and 10.1 exports validation-stage records (port of
  FRR1-m3). Reproduced before the fix on CPU with stub state: 8.1 `ValueError`, 8.3 and 8.4 `RuntimeError`.
- **REG2-m5:** 6.1 removes only its own classical keys, after the development check. Reproduced before the fix: a
  rerun dropped the 6.2 key, and in a frozen experiment it emptied the list before refusing.
- **REG2-m3, m4, m6, m8–m11, S1:** plots follow `PRIMARY_METRIC`; research question 2 is an optional GPU extension; the
  6.2 activity uses the month/weather group that Run all does not print; the checkpoint-selection metric (validation
  MAE, fixed in the carried code) is stated; total runtime (651.9 s, Colab T4 at `aca2655`) and computed download size
  (about 14 GB); model papers and pinned cards; a Troubleshooting row for the frozen-experiment error.

Verification is offline only (`tests/test_workshop_v2_reg2_review_fixes.py`, CPU, stub state); it is not
clean-runtime evidence. A hosted Run all of the new blob is still required.

### Peer review of the REG2 fixes and owner decisions — 2026-10-14

- **Spec:** the guided edition now declares NOTEBOOK_SPEC 2.5 (metadata, opening, manifest, completion summary,
  validator). The revision label stays 3.2.0; bumping both editions to 3.2.1 in one PR is a follow-up.
- **Evaluation status:** 8.0 records `evaluation_status` (`confirmatory` only when no earlier test exposure is recorded,
  otherwise `exploratory (test previously exposed)`); 8.2, 8.3, the export manifest and the conclusion template use it.
- **Exposure logging:** logged before any test-target display (3.1 and 3.6) and before test scoring
  (`test_scoring_attempted`, then `test_scoring_completed`), so a failure part-way through still counts.
- **Wording:** restarting restores runtime state only; issuance after day *t* closes; pooled, row-weighted scope on
  the same sampled series; prediction-day stockout labels; illustrative-IID plot titles and glossary; R² of the
  training-mean predictor; an ablation-versus-parent point-difference table in 8.3; BYOD protocol declared as
  unverified and extra feature columns refused unless `ALLOW_ADDITIONAL_FEATURES` is set.

### Colab CLI execution of guided v2 at `4c44ff0` (blob `dfb253c68e55`) — 2026-10-14

- **Executor:** Colab CLI 0.7.4 sequential execution on a fresh Colab Tesla T4 VM (`colab exec -f`, workspace
  `colab-cli-serial-test-suite`): every code cell in order in one kernel, order taken from `exec.log` ("Executing cell
  k/36"). Not a browser **Run all**; no execution counts; forms and uploads not exercised.
- **Notebook:** `tutorials/DIMER_FreshRetailNet_MultiModel_Regression_Workshop_v2.ipynb`, commit
  `4c44ff086c3a9f6c2d3c39c2667b36b5d1e6225c`, blob `dfb253c68e55ce7860ecdf6d5a54bb806e8266d6` (fetched byte-exact at the
  commit and verified before the session; the executed copy's 88 cells have the same ids and order and its code equals the
  source).
- **Run:** 2026-10-14 23:17:38–23:27:24 UTC, session wall 593.9 s. Kernel Python 3.13.15 on Linux x86_64; section 0.1
  installed nothing. All four uv environments were built from the hash-locked files and all four foundation models ran on
  `cuda`.
- **Outcome:** **PASSED** — 36/36 code cells, 0 error outputs; the completion summary reports `notebook_spec` 2.5, 9
  validation models, 12 frozen test models, no foundation failures, and "Run-all complete".
- **REG2 records on a single default run:** 8.0 printed the validation-selected model `tabiclv2_icl` (TabICLv2, best
  validation MAE 0.36674) with evaluation status `confirmatory`; in 8.2 and 8.3 its row is the only `confirmatory` row and
  the other 11 are `descriptive`; 8.3 prints its test MAE 0.43286 as the confirmatory result. No row or line reports
  `exploratory (test previously exposed)`, so no earlier test exposure was recorded in the runtime. 8.3 also prints the
  illustrative-IID and multiplicity notes and the ablation-versus-parent table (LightGBM without month and weather
  -0.00142, LightGBM without stockout +0.00338, TabICLv2 without stockout +0.01600).
- **Comparison with the `aca2655` run (blob `5ce1e3e1`, 2026-10-03):** every printed metric is identical — the baseline
  and validation tables, the ablations, the 7.1 error bands, the frozen-test MAE / RMSE / median absolute error / R²,
  the bootstrap intervals, gains and shares, the 8.4 prediction preview and the 9.1 band scores. The differences are:
  the descriptive `runtime_seconds` column (for example Mitra 95.81 s → 90.12 s, TabICLv2 10.59 s → 8.80 s); run ids,
  report-directory names and the report ZIP digest (new experiment directories, by construction); uv download order
  and timings, AutoGluon memory and throughput lines; one Hugging Face "unauthenticated requests" warning that this
  run did not print (environment output, not touched by the PR); and the additions made by the REG2 fixes — the
  `validation_selected` and `evaluation_status` columns, the validation-selected lines, the IID and multiplicity notes,
  the ablation-versus-parent table and `notebook_spec` 2.1 → 2.5. This confirms the static assessment that no
  default-path model or metric computation changed.
- **Evidence file** (`docs/execution-evidence/2026-10-14/`, byte-exact, covered by the `-text` rule):
  `DIMER_FreshRetailNet_MultiModel_Regression_Workshop_v2_4c44ff0_colab-cli-t4.ipynb`, SHA-256
  `293ce0bb5ceb0b3a4079ed9ae8f1177780268dbef11bd55200e101405bec2072`.
- **Evidence boundary:** CLI sequential execution, not a browser Run all. Not exercised: the interactive activities, the
  **Freeze now** off path, Start new experiment after test exposure (the `exploratory` status path), a foundation-only
  selection, R² as the primary metric, BYOD, the fast path and fine-tuning.

This record satisfies the hosted default-path run of the REG2 blob. The other open items remain. **Status: Candidate.**

### Local-path redaction of earlier evidence (2026-10-22)

The local run directory in these earlier committed evidence files was replaced by `<local-run-dir>/`, the
convention the later evidence in this repository already uses. Only those bytes changed: line endings and every
other byte are as committed. Git history is not rewritten, so earlier commits still hold the original files.
SHA-256 before (committed on `main`) → after:

| File | Replacements | SHA-256 before | SHA-256 after |
| :--- | ---: | :--- | :--- |
| `docs/execution-evidence/2026-10-08/mitra_regressor_colab/exec.log` | 2 | `43a346373ac5d43f481c933caf8a7723fcc4cb9dc0513bc8167f24f1a3ca316b` | `8a384d833e71bb2a1060c3053b4e737f7995a2c1f5ececa16a522064793462da` |
| `docs/execution-evidence/2026-10-08/mitra_regressor_predictor_inference_colab/exec.log` | 2 | `bdfd37432715114e72f600386753e9ce868584b5b2dc79f95301c00bd829779b` | `a7555e02ac44410ae62303a3b1fb1685751db79f5aaaac4c9116f2415c9dee1f` |

Where this document quotes one of these files' earlier hash as its committed hash, the new hash is now
given beside it.

