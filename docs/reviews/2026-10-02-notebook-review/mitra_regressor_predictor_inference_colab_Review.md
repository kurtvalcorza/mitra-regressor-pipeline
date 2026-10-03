# Mitra Regressor Predictor-Inference Notebook — Review

**Verdict: Needs revision**  
**Review date:** 4 October 2026 (relay batch of 2 October 2026)  
**Repository:** `kurtvalcorza/mitra-regressor-pipeline`  
**Notebook:** `tutorials/mitra_regressor_predictor_inference_colab.ipynb`  
**Reviewed commit:** `dc2d022e62eba4c6c9a25e61c2dbb785e0a77809` (`main`, confirmed with `gh api repos/kurtvalcorza/mitra-regressor-pipeline/commits/main`)  
**Notebook Git blob:** `7cbd9f31095bc8d5a7874cb3a5f9ded8e70ebd84` (last changed in `7a4efc7`, 2026-09-14)  
**Finding prefix:** `MRP` (the companion E2E notebook is `MRC`, reviewed in its own row; the FreshRetailNet workshops use `FRR1`/`REG`)  
**Framework:** Notebook Review Framework v1. **Requirements baseline:** NOTEBOOK_SPEC 2.2 (2026-09-26), `ml-worker` `origin/main` `50b7eb4`. The notebook declares 2.0.

## Executive assessment

The external-artifact machinery works. Given a bundle path, its trusted digest and a CSV path, all seven code cells ran on CPU in a fresh process. The cells verified the whole-archive SHA-256, extracted the bundle safely and checked its manifest, provenance and base-model identity before `TabularPredictor.load`. They then reconstructed a regression predictor, validated 89 new rows into an input manifest with a recorded rejection probe, and exported four files. All 89 predictions are finite floats in target units. Five invalid inputs were refused before deserialisation or model execution, each with a message that names the failed condition. The trust-boundary prose is accurate: digests prove integrity, not sender authenticity or safe unpickling. The prose on point estimates without uncertainty is also accurate.

Unlike its classifier twin, this notebook has hosted execution evidence for the external-artifact path. The repository's `Notebook release execution` workflow ran this exact notebook blob with the three path fields set and passed (run `37101095191`, head `6d17c71`, Ubuntu 24.04, Python 3.12.14, nbconvert). That run is not recorded in `docs/release-verification.md`, and the rows it scores are partly the bundle's own support rows (MRP-m6).

The notebook still cannot do what its profile promises by default:

1. **The default `Run all` has no sample artifact and no sample input (MRP-B1).** With the forms untouched, Section 4 opens an upload dialog on Colab. On Jupyter it raises `ModuleNotFoundError: No module named 'google'`. Even after a learner uploads a valid bundle, cell 9 raises because `EXPECTED_ZIP_SHA256` is empty by default. The notebook's own opening says so.
2. **Supplying the bundle by path and the rows by upload crashes (MRP-M1).** The upload branch in cell 13 calls `files.upload()`, but `files` is imported only inside cell 9's upload branch. This is the combination the B1 fix will create, because the sample bundle will come by path and BYOD rows by upload.
3. **The declared `GUIDED` layer is absent (MRP-M2).** There are no expected-result notes after Section 1, no prediction prompts, checkpoints, glossary or conclusion template, and no in-notebook activity. The 1,035-line carrier cell is neither labelled nor collapsed.

These three findings are the same defects found in the classifier twin (MCP-B1, MCP-M1, MCP-M2, PR #46) and are graded the same way.

## 1. Review contract and evidence

| Item | Value |
|---|---|
| Declared profile / mode | `ARTIFACT-INFERENCE` / `GUIDED` (`metadata.dimer`, opening cell) |
| Declared spec | DIMER Notebook Specification **2.0**, standalone (§4) |
| Spec baseline applied | NOTEBOOK_SPEC **2.2**; IDs below are 2.2 IDs |
| Generator | `tools/build_notebook.py` (build_notebook.py/2) with template `tools/notebook_template_artifact_inference.py`. Carried module `mitra_pipeline/tutorial_api.py`, sha256 `ad5f2f3a…` (equal to `HEAD`). `build_notebook.py --template … --check` → `OK … is up to date` |
| Intended audience | Not stated. The prerequisites list the runtime, the artifact, the data and external access, but not the learner's assumed level |
| Supported runtime | "Google Colab or Jupyter, Python 3.10–3.13". The default path is CPU, with CUDA used if present. The bundle's AutoGluon version and Python major/minor must match the runtime |
| Promised outcomes | Externally produced bundle → whole-archive digest → safe extraction → manifest/provenance/base-model checks before deserialisation → runtime-compatibility check → reconstruction from the bundle alone → validated new rows with an input manifest → continuous point predictions (no uncertainty) → `not-measurable` evaluation report → machine-readable export. BYOD for new input. Optional user artifact |
| Cells | 17 cells: 7 code, 10 markdown. Code cells 3, 5 (1,035-line, 44.9 kB carrier), 7, 9, 11, 13 and 15. No persisted outputs; all code cells compile |

**Existing execution evidence**
- **CI (documented, not Colab):** `.github/workflows/notebook-release.yml` runs the E2E notebook, copies its bundle, and executes this notebook in a copy with `ARTIFACT_ZIP_PATH`, `EXPECTED_ZIP_SHA256` and `NEW_DATA_PATH` set. The latest run, `37101095191` (2026-10-03, head `6d17c71` = second parent of `main`), passed every step. The notebook blob at that head is `7cbd9f3…`, the reviewed blob. Its evidence artifact (280 MB) is retained until 2026-11-02 and was not downloaded for this review. The run executes the path journey only, not the default `Run all`.
- **Manual:** `docs/release-verification.md` lists the artifact-inference row as "pending — queued to the GPU lane". `tutorials/README.md` lists the notebook as **Candidate** with "unverified — no clean-runtime `Run all` execution recorded".
- **Colab:** none for any revision.

**Journeys and evidence basis**

| Journey | Basis | What was done |
|---|---|---|
| First-time learner | Source inspection | All 17 cells read in order against GDL/UX/§19 |
| Clean default | Direct execution (CPU, fresh process, no form edited) | P1, plain Jupyter with no `google.colab`: cell 9 → `ModuleNotFoundError`; cells 11/13/15 → `NameError`. P2, Colab upload stubbed with a valid bundle and the default digest field: cell 9 → `RuntimeError: A trusted whole-archive SHA-256 is required…`. Colab browser `Run all` **not verified** (no sample asset exists to make it pass) |
| External-artifact path (`release-verification.md` step 6) | Direct execution (CPU) + documented CI execution | P3: bundle produced by a **separate process** that ran the companion E2E notebook at the same commit (diabetes sample, ZIP 280,010,397 bytes, sha256 `831ab78d…`), plus the 89 unlabelled rows of that run's independent test partition. With the three path fields set, cells 3–15 all pass (cell 15: 47.9 s on CPU) and four outputs are written. Reviewer-side MAE against the withheld labels was 44.9, against 66.5 for predicting the test mean; the notebook reports nothing because it has no labels. CI run `37101095191` passed the same journey on Linux |
| Active learning | Not applicable within the notebook / partly probed | The notebook has no exercise; its "Next experiments" send the learner to the E2E notebook (MRP-M2). Closest probe, R10: a labelled CSV is accepted, `target` passes through as an extra column, and the verdict stays `not-measurable` (MRP-S2) |
| Reuse and recovery | Direct execution (CPU) | P4: artifact by path + rows by upload (stub) → `NameError: name 'files' is not defined` (MRP-M1). P5 refusals: wrong digest, non-hex digest, unlisted member (digest re-trusted), `../` traversal, CSV missing a feature, CSV with a `prediction` column. All are refused with the condition named. `ALLOW_UNVERIFIED_ARTIFACT=True` passes with its warning. A one-row CSV predicts. A non-numeric feature value passes validation and fails inside AutoGluon (MRP-m4). P6: the bundle reconstructs and predicts **offline with an empty HF cache and without Section 3** (MRP-m1). Python/AutoGluon mismatch was read in source only |

**Deviations and limitations**
- **Host and environment.** Windows CPU host, using `mitra-regressor-pipeline/.venv` read-only. It holds the exact pins: autogluon.tabular 1.5.0, lightgbm 4.6.0, huggingface-hub 0.36.2, Python 3.12.12, torch CPU. The install cell was skipped with `DIMER_NOTEBOOK_CI_PREINSTALLED=1`, so the restart guard in cell 3 was not exercised.
- **Memory override, labelled.** The bundle producer, the companion E2E run, set its own form control `MAX_MEMORY_USAGE_RATIO=3.0` because the host had only about 3.5–7 GB of free RAM. The notebook under review needed no override.
- **Weights.** The Mitra snapshot was copied in from Kurt's checkout, with digests equal to the manifest. The Hub download was not exercised.
- **Stubs.** `google.colab.files.upload` was stubbed in P2 and P4 only. The real Colab dialog, including uploading a 280 MB file through the browser, was not exercised.
- **No GPU run.** The CI evidence artifact was not downloaded; the CI outcome is taken from the run's step conclusions.

## 2. Separate judgments

- **Technical correctness.** The core path is correct when files are supplied by path, here and in CI. The checks run in the stated order, before any deserialisation (`validate_artifact_directory` reads only JSON). Predictions are finite floats, and outputs carry every input column. There are defects. The default path cannot complete (MRP-B1). `files` is used across cells as hidden state (MRP-M1, SRC2). `serving` depends on Section 3's `pipe` although the prose says it does not (MRP-m1). Feature dtypes are not validated (MRP-m4).
- **Promise fulfilment.** Every artifact-handling promise holds on the path journey. "Reconstruct the predictor from the bundle alone" was confirmed directly (P6). The profile's central promise fails: "the default `Run all` automatically obtains a trusted sample artifact and sample input" (§7.4/§19). The notebook says so, but saying so does not satisfy a MUST. The claim that the carried module is "at revision `b249861b89bd`" is wrong: those bytes are from `7a4efc7` (MRP-m5).
- **Learner experience.** The prose is accurate but written for an engineer or reviewer. The guided layer is missing, and the only activity sends the learner to another notebook (MRP-M2). Upload-based inputs assume Colab and a practical 280 MB browser upload (MRP-m3).
- **Spec conformance.** These applicable MUSTs are **unmet**: SART1, RUN1, RUN2, RUN5, DAT1, DAT3, the §19 bullets for automatic sample artifact, no default upload and automatic sample input, REL1/REL2/REL4 (the CI evidence covers the path journey, not the default `Run all`), and VAL2/DAT19 for feature types (MRP-m4). These MUSTs are **met** on the path journey: §20 archive security (traversal, unlisted-file rejection and the whole-archive digest were probed; symlink, size and ratio checks were read in source), the §19 trust boundary, validation before deserialisation, display of model identity, format version and provenance, no refit from inference data, MOD1–MOD3, MOD7, INF3 and INF5, DAT17/DAT18, VAL4 and OUT6. These SHOULDs have unrecorded deviations: GDL1–GDL14, UX5/UX8 (MRP-M2) and EXE2 (MRP-M1/m3). VAL6 (a MUST) is met only for training-side ceilings, which are mislabelled (MRP-m2).

## 3. Findings

### MRP-B1 (Blocker): the default `Run all` has no trusted sample artifact or sample input

- **Cell/section:** opening cell 0 ("Known NOTEBOOK_SPEC 2.0 gap"); prerequisites cell 1; Section 4, cell 9; Section 6, cell 13.
- **Observed issue:** `ARTIFACT_ZIP_PATH = ''` and `NEW_DATA_PATH = ''` fall through to `from google.colab import files; files.upload()`. `EXPECTED_ZIP_SHA256 = ''`, together with `ALLOW_UNVERIFIED_ARTIFACT = False`, makes cell 9 raise even after a valid upload. Nothing in the notebook obtains a bundle or rows. The prerequisites say "no sample is bundled".
- **Consequence:** A learner who opens the notebook and chooses **Run all** cannot complete it. On Colab, Run all stops at a dialog and needs a ~280 MB bundle produced in another session, plus its digest pasted into a field. On Jupyter it fails outright. The profile's default demonstration never runs, and no default-path evidence can be recorded. The CI workflow works around this by rewriting the three fields in a copy.
- **Evidence:** Direct execution: P1 (Jupyter: `ModuleNotFoundError: No module named 'google'` at cell 9, then `NameError`s) and P2 (valid bundle uploaded, default digest field: `RuntimeError: A trusted whole-archive SHA-256 is required…`). Source inspection: cells 0 and 1, `notebook-release.yml` field rewriting. Documented: `tutorials/README.md` status "Candidate", default sample "**none yet**".
- **Spec:** SART1, SART2, SART5, RUN1, RUN2, RUN5, DAT1, DAT3, §7.4, §19 bullets 1, 2 and 7, REL1, REL2, REL4.
- **Recommended correction (generator: `tools/notebook_template_artifact_inference.py`, the cell-9 source at lines ~95–115, the cell-13 source at ~175–185, and the opening text at line 26):**
  1. Publish one trusted sample bundle that the companion E2E notebook produced at a pinned revision on the diabetes sample. Use a stable public, credential-free location with an immutable reference, for example a Hugging Face Hub repo at a commit SHA or a GitHub release asset. ST6/SART2 allow a downloaded trusted sample artifact; ST4 forbids only DIMER source.
  2. Make the default branch download it, with its SHA-256 as the default trusted digest, before `safe_extract_archive`. State its provenance, format version and digest in markdown before deserialisation (SART3).
  3. Generate the default sample input in-notebook with `sklearn.datasets.load_diabetes`, restricted to rows disjoint from the bundle's support rows. For example, reproduce the E2E notebook's seeded test partition and assert the overlap with the support rows is zero (see MRP-m6 for why this matters).
  4. Gate both uploads behind default-off switches (`USE_OWN_ARTIFACT`, `USE_BYOD`).
  5. Because AutoGluon bundles are bound to the Python major/minor, record the sample bundle's producer Python. Document that the bundle must be rebuilt when Colab's Python changes, or the default path fails in cell 11.
  6. Set `sample_kind` to `sample` on the default path (it is hard-coded `BYOD` in cell 15).
  7. Remove the "Known gap" paragraph and update `tutorials/README.md`.
- **Acceptance check:** In a fresh Colab runtime with no form field edited, **Run all** completes all code cells with no dialog and no error. `outputs/` holds the four `mitra_regressor_predictor_inference_*` files. `result.json` has `artifact.source` naming the published sample location and `digest_verified: true`, and `evaluation_report.sample_kind` is `sample`. The run is recorded in `docs/release-verification.md` with commit, blob and runtime (REL10).

### MRP-M1 (Major): the CSV upload branch raises `NameError` whenever the bundle came by path

- **Cell/section:** Section 6, cell 13 (`new_upload = files.upload()`).
- **Observed issue:** `files` is bound only by `from google.colab import files` inside cell 9's `else:` branch. When `ARTIFACT_ZIP_PATH` is set and `NEW_DATA_PATH` is empty, cell 13 uses a name that was never defined. This is a hidden cross-cell dependency.
- **Consequence:** An executor or learner who supplies the bundle by path and the rows by upload hits `NameError` with no recovery message. After MRP-B1 is fixed, the default sample bundle will always come by path, so this becomes the BYOD-upload path for every learner. The CI workflow never reaches this branch because it sets both paths.
- **Evidence:** Direct execution, P4: cells 3–11 pass with the artifact by path and digest; cell 13, with the Colab stub holding one CSV, raises `NameError: name 'files' is not defined`. Static probe: `files` imported in cell 9 only, used in cells 9 and 13.
- **Spec:** SRC2, EXE2, DAT10/DAT15, UX10.
- **Recommended correction:** Import `from google.colab import files` inside cell 13's upload branch (template line ~179). Better, follow the B1 gating so that each upload branch imports its own dependency and fails with an actionable message outside Colab.
- **Acceptance check:** P4 (`run_probes.py mixed`) completes cell 13 and cell 15 with no error, and an equivalent Colab run with the artifact by path and the CSV by dialog writes the four outputs.

### MRP-M2 (Major): the notebook is declared GUIDED, but the guided layer and any learner activity are absent

- **Cell/section:** whole notebook; opening cells 0–1; Sections 3–7; cell 16 "Interpretation and limits"; carrier cell 5.
- **Observed issue:**
  - No intended learner or assumed level (GDL1), "How to use this notebook" (GDL2), roadmap (GDL3) or Input → Model → Output contract (GDL4).
  - The 11 "learning objectives" are pipeline steps ("install the pinned runtime", "read what the carried package guarantees"), not observable learner outcomes (GDL5/UX1).
  - No glossary for predictor bundle, whole-archive digest, manifest, deserialisation, support rows or point estimate (GDL6).
  - Only Section 1 says what to look for; Sections 3–7 have no expected-result notes (GDL8/UX4), and no prediction prompts or checkpoints with sample answers (GDL7/GDL9). Static probe: none of "expected result", "what to notice", "glossary", "troubleshoot", "how to use", "infrastructure", "exercise" or "conclusion" appears. "checkpoint" and "predict" appear only as model and output words.
  - No Predict → Change → Run → Observe → Explain activity (GDL10/UX5). Both "Next experiments" require the E2E notebook.
  - The predictions table (`out.head()`) gets no reading guidance: nothing says what a plausible range is for the diabetes target, or why there is no interval.
  - No conclusion template (GDL14). The failure table covers integrity failures but not upload size, Python/AutoGluon mismatch with the current Colab, memory, or non-numeric features (GDL13).
  - The 1,035-line, 44.9 kB carrier cell is not labelled Infrastructure or collapsed (GDL11/GDL12).
- **Consequence:** The learner can run the cells but is not taught to read a predictor bundle's provenance, to judge what an uncalibrated point estimate means, or to decide what to do with a refusal. The intended audience is undefined, so the dense engineering prose cannot be checked against it.
- **Evidence:** Source inspection of all 10 markdown cells; static marker probe (`S_static.markers`).
- **Spec:** GDL1–GDL14, UX1, UX4, UX5, UX8, UX9.
- **Recommended correction (template text in `tools/notebook_template_artifact_inference.py`):**
  - Add an audience and how-to-use block and an I/O contract.
  - Rewrite the objectives as observable actions, for example "explain which checks run before deserialisation and why".
  - Add an expected-result note after Sections 3–7.
  - Add one bounded in-notebook activity on the sample. For example, predict and then observe how a tampered bundle (an extra file) or a CSV missing one feature is refused, or compare predictions for two rows that differ in one feature. A labelled copy of the sample input would also allow a sample-sanity report (MRP-S2).
  - Add a glossary, a troubleshooting section and a conclusion template.
  - Label and collapse the carrier as Infrastructure.
- **Acceptance check:** A reviewer finds each of GDL1–GDL14 addressed in the regenerated notebook, or each deviation recorded in `tutorials/README.md`. At least one activity runs inside this notebook on the default sample without editing more than one form field, and does not break **Run all**.

### MRP-m1 (Minor): Section 5 says Section 3's `pipe` is not used for inference, but cell 11 builds `serving` from it; Section 3's stated reason is inaccurate

- **Cell/section:** cell 10 ("The pinned snapshot `pipe` of Section 3 is not used for inference"); cell 11 (`serving = MitraRegressionPipeline(weights_path=pipe.model_weight_path, config_path=pipe.config_path, snapshot_path=pipe.snapshot_path, device=pipe.device, …)`); cell 0 ("it exists so the bundle's recorded base-model digests can be checked against known-good values"); Section 3, cell 7.
- **Observed issue:** Predictions go through `predictor` only (`predict_regression` calls `predictor.predict`). The bundle carries `models/Mitra/model.pt` (302,787,855 bytes) and reconstructs and predicts with an empty HF cache, offline, without Section 3 (P6: 89 rows, 21.6 s, no cache files written). Cell 9 compares the bundle's metadata digests with the carried constants `WEIGHTS_SHA256`/`CONFIG_SHA256`, not with the downloaded file. Yet cell 11 reads `pipe.*`, so skipping Section 3, as the prose implies is safe, raises `NameError`.
- **Consequence:** A fresh runtime downloads about 303 MB and stages extra copies for no inference purpose (RUN12). The learner is told the download is needed for a check it does not perform, and is told `pipe` is unused when it is a hidden dependency (SRC2).
- **Evidence:** Direct execution (P6; P3 `serving_model_weight_path_is_section3_pipe: true`); static probe (`pipe_not_used_claim_in_md` and `pipe_used_in_cell11` both true); source inspection of `predict_regression` and `validate_artifact_directory`.
- **Spec:** RUN12, SRC2, UX2, ENV9.
- **Recommended correction (template lines ~142 and the cell-11 source):** Either drop the snapshot staging from this profile and build `serving` from the bundle and provenance alone, or keep Section 3 as an explicit optional step, say plainly that the base-model check uses the pinned constants, and stop claiming `pipe` is unused.
- **Acceptance check:** The regenerated notebook either completes Run all without fetching `model.safetensors`, or its prose states what Section 3 adds and no cell after Section 3 reads `pipe` while the markdown says it is unused.

### MRP-m2 (Minor): inference ceilings are not surfaced; training ceilings are printed instead

- **Cell/section:** Section 6, cell 13 (`print({'ceilings': {'MIN_TRAIN_ROWS', 'MAX_TRAIN_ROWS', 'MAX_FEATURES'}…})`) and the input manifest's `schema` block.
- **Issue → consequence:** The printed ceilings govern the producer's fit, not this stage. `validate_inference_frame` enforces no row or size limit, so a large CSV is read fully into memory and scored without warning. A learner reading "ceilings" may assume a 10,000-row inference limit that does not exist. The written inference manifest also describes a numeric target and `train_rows` [50, 10000], which do not apply to unlabelled rows.
- **Evidence:** Source inspection (`eval_rows` [2, None]; no inference cap in `validate_inputs`). Direct execution: a one-row CSV was accepted and predicted (R9); P3 manifest text.
- **Spec:** VAL6, DAT12.
- **Correction:** Print the inference contract (required features and their expected types, reserved columns, no row cap or an explicit `MAX_INFERENCE_ROWS`) and label the training ceilings as producer-side context.
- **Acceptance check:** Cell 13 prints the inference-stage limits under an inference label before `read_csv_bytes`.

### MRP-m3 (Minor): the upload branches assume Colab, and a browser upload of a ~280 MB bundle is impractical

- **Cell/section:** cell 1 ("Google Colab or Jupyter"); cells 9 and 13 (`from google.colab import files`).
- **Issue → consequence:** On Jupyter, which the notebook names as a supported runtime, both upload branches fail with `ModuleNotFoundError` and no guidance. On Colab, the user-artifact branch asks for a ~280 MB ZIP through `files.upload()`, which is slow and is held in memory twice (payload plus written copy). No mounted-storage option is mentioned.
- **Evidence:** Direct execution, P1 (Jupyter). Measured bundle size 280,010,397 bytes (B_bundle). The Colab dialog itself was not verified.
- **Spec:** EXE2, UX10, GDL13, DAT16.
- **Correction:** Guard the upload branches with an actionable message ("set `ARTIFACT_ZIP_PATH`/`NEW_DATA_PATH` outside Colab"). Recommend a path or Google Drive mount for user artifacts, and mention the expected size.
- **Acceptance check:** With no `google.colab`, setting `USE_OWN_ARTIFACT=True` and leaving the path empty raises a message naming the path field. The prose names a non-upload route for bundles.

### MRP-m4 (Minor): non-numeric feature values pass validation and fail later inside AutoGluon

- **Cell/section:** Section 6, cell 13 (`validate_inputs` / `validate_inference_frame`); Section 7, cell 15 (`serving.predict`).
- **Issue → consequence:** The inference schema accepts "feature columns of any dtype". A CSV whose `age` column holds `abc` is written to the input manifest with no finding and fails only at prediction, with `ValueError: could not convert string to float: 'abc': Error while type casting for column 'age'`. The message names the column, but it comes from inside the model after the manifest has recorded the input as valid, and the recovery table does not cover it.
- **Evidence:** Direct execution, R11 (cell 13 passes; cell 15 raises). Source inspection of `validate_inference_frame`.
- **Spec:** VAL2, DAT19 (MUSTs: types checked explicitly; failures must identify the contract rather than fail later inside model code), VAL1.
- **Correction:** Record each feature's training dtype in the bundle metadata (or read it from the predictor's feature metadata after the compatibility checks) and check it in `validate_inference_frame`, rejecting values that cannot be coerced with a message naming the column and the expected type. Add the row to the failure table.
- **Acceptance check:** R11 (`run_probes.py recovery`) raises in cell 13, not cell 15, with a message naming `age` and its expected numeric type, and the input manifest is not written as accepted.

### MRP-m5 (Minor): the declared source revision does not hold the carried module bytes

- **Cell/section:** cell 0 ("carries … `mitra_pipeline/tutorial_api.py` at revision `b249861b89bd` verbatim"); cell 3 `NOTEBOOK_SOURCE.repository_revision`; `metadata.dimer.generated_from.revision`; Section 2 heading.
- **Issue → consequence:** The carried module's sha256 is `ad5f2f3a…`, recorded correctly in `module_sha256`. At `b249861` the module's sha256 is `a6b564ce…`; the carried bytes first appear in `7a4efc7`. The generator records `HEAD` at generation time as a "provenance label", but the learner-facing text says the module is "at revision" `b249861` "verbatim". Anyone who checks out `b249861` to audit the carried code gets different code, and `result.json` exports the same mislabelled revision.
- **Evidence:** Static probe (`S_static.provenance`); source inspection of `_head_revision` in `tools/build_notebook.py` (lines 266–276). The companion E2E notebook carries the same label.
- **Spec:** OUT3 (provenance), ST5 (logic visible and traceable); no MUST is violated, because the module digest is exported.
- **Correction:** Record the last commit that touched the carried modules (or regenerate after committing the module), or reword the opening and Section 2 to say "module sha256 … (generated while `HEAD` was …)".
- **Acceptance check:** `git show <declared revision>:mitra_pipeline/tutorial_api.py | sha256sum` equals the notebook's `module_sha256`, or the notebook no longer claims the module is "at revision" the declared commit.

### MRP-m6 (Minor): the CI release run passes but is not recorded, and the rows it scores overlap the bundle's support rows

- **Cell/section:** `.github/workflows/notebook-release.yml` step "Prepare independent inference input"; `docs/release-verification.md` "Recorded executions"; `tutorials/README.md` row for this notebook.
- **Issue → consequence:**
  - Run `37101095191` executed this notebook blob against an external E2E bundle and passed. `release-verification.md` still says "pending", and the README says "unverified". The repository understates its own evidence, and nothing records the runtime and outcome as REL10 asks.
  - The workflow's "independent inference input" is `load_diabetes().data.tail(24)`. All 24 rows are rows the E2E notebook partitioned: 12 are in the support set the bundle conditions on, 6 are in the holdout and 6 are in the test partition. The notebook promises "genuinely new unlabelled rows", but the recorded run scores rows the predictor already holds, so it cannot be cited as new-data evidence.
- **Evidence:** Documented: run step conclusions and head SHA (`gh run view`); notebook blob equal at `6d17c71` and `dc2d022`. Direct execution: `B_bundle.ci_workflow_tail24_overlap` (same seed and split as the E2E default).
- **Spec:** REL10, REL8 (keep static and execution evidence distinct), §19 "genuinely new" input, DAT8.
- **Correction:** Generate the CI inference rows from the E2E run's test partition (or assert zero overlap with the bundle's support rows), and record each passing run in `release-verification.md` as CI execution of the path journey (commit, blob, runner, Python, outcome), separate from the still-missing default `Run all` evidence.
- **Acceptance check:** The workflow asserts zero overlap between the scored rows and the bundle's support rows, and `release-verification.md` has a row naming a passing run ID, the commit, the notebook blob and the runner.

### Suggestions

- **MRP-S1:** Regenerate against NOTEBOOK_SPEC 2.2 (metadata and opening declare 2.0), and use the §29 artifact-inference opening template.
- **MRP-S2:** When the scored CSV contains the bundle's `target_column` (R10: it passes through as an extra column and the verdict stays `not-measurable`), offer an optional labelled check with `regression_metrics` and `training_mean_baseline` in this notebook, labelled sample-sanity. That would let the first "Next experiment" run here instead of in the E2E notebook.
- **MRP-S3:** Show the learner a compact provenance summary from the bundle (data source, support rows, mode, selection basis, target units) next to the predictions, so that "what was this predictor trained on?" is answered where the predictions are read.

## 4. Readiness

**Needs revision.** One Blocker (MRP-B1) and two Majors (MRP-M1, MRP-M2) are open. Applicable MUSTs fail: SART1, RUN1, RUN2, RUN5, DAT1, DAT3, the §19 default-path bullets, REL1/REL2/REL4 and VAL2/DAT19. The CI run covers the external-artifact path only; no default `Run all` evidence exists for any revision.

Remaining gates, in order:
1. Fix MRP-B1 and MRP-M1 in the generator and regenerate; `--check` and the parity tests must stay green.
2. Address MRP-M2, or record each GDL deviation durably.
3. Fix MRP-m4 (a MUST) and make the CI input rows disjoint from the support rows (MRP-m6).
4. Run a fresh Colab `Run all` with no field edited and record it in `docs/release-verification.md`, alongside the CI path-journey runs.
5. Separately exercise the user-artifact and BYOD branches (REL12).

The remaining Minors and Suggestions do not gate release.

## 5. Verified versus inferred

- **Verified by direct execution (CPU, this host, exact pins):**
  - The default path fails on Jupyter (P1), and with an uploaded bundle and the default digest field (P2).
  - The path journey passes end to end, with four outputs and finite predictions (P3).
  - The mixed path raises `NameError` (P4).
  - Six refusals, the override warning, and a non-numeric feature accepted by validation then rejected inside AutoGluon (P5).
  - The bundle reconstructs offline without Section 3 (P6).
  - The CI workflow's 24 rows overlap the E2E partitions, 12 of them the support set (B_bundle).
  - The generator `--check` is clean; the declared revision's module bytes differ from the carried bytes.
- **Verified from documented evidence:** CI run `37101095191` passed every step on the reviewed notebook blob (step conclusions only; the executed notebook was not downloaded).
- **Verified by source inspection:** check order before deserialisation; the trust-boundary prose; the absence of the guided layer; the ceilings printed; the `pipe` contradiction.
- **Inferred / not verified:**
  - Real Colab behaviour, including the upload dialog and a 280 MB browser upload.
  - The Hub download in Section 3.
  - The install cell's restart guard. It still pip-installs into the kernel, so if a Colab run shows a restart is needed, the fix should use the uv isolated environment (`bioclip2-biodiversity-pipeline/tutorials/DIMER_Philippine_Biodiversity_Field_Survey_Capstone.ipynb`), not a new guard. No restart was observed in CI or in the Kaggle E2E record.
  - GPU execution.
  - A Python/AutoGluon-mismatch bundle (read in source only).
  - Learner understanding (no learner observation).
- **Only Kurt can confirm:** where a public sample bundle may be hosted, and whether the 280 MB diabetes bundle is acceptable as the published sample.
- **Finding most likely to be wrong:** MRP-M1's severity. Today it bites only when an executor mixes path and upload, so it could be argued Minor. It is graded Major, as MCP-M1 was, because the MRP-B1 fix will make "bundle by path, rows by upload" the standard BYOD route for every learner.

*Probe bundle:* `mitra_regressor_predictor_inference_colab_Review_Probes.zip` (`run_probes.py`, `results.json`, `source_manifest.json`).
