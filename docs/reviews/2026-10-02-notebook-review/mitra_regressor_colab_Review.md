# Mitra Regressor E2E Tutorial Notebook — Review

**Verdict: Needs revision**  
**Review date:** 4 October 2026 (relay batch of 2 October 2026)  
**Repository:** `kurtvalcorza/mitra-regressor-pipeline`  
**Notebook:** `tutorials/mitra_regressor_colab.ipynb`  
**Reviewed commit:** `dc2d022e62eba4c6c9a25e61c2dbb785e0a77809` (`main`, confirmed with `gh api repos/kurtvalcorza/mitra-regressor-pipeline/commits/main`)  
**Notebook Git blob:** `d32d6f6b45c6d3726c0f567cd9a993a68f526b56`. This is the blob executed in the recorded Kaggle Tesla T4 run of 2026-09-14 (commit `7a4efc7`, kernel `dimer-nb2-mitra-regressor` v2).  
**Finding prefix:** `MRC` (the FreshRetailNet workshops use `NR`/`FRR1`/`REG`; the companion inference notebook is a separate row)  
**Framework:** Notebook Review Framework v1. **Requirements baseline:** NOTEBOOK_SPEC 2.2 (2026-09-26), `ml-worker` `origin/main` `b1cfe13`. The notebook declares 2.0.

## Executive assessment

The default path is well engineered and runs. It installs three exact pins, carries `mitra_pipeline/tutorial_api.py` verbatim, digest-verifies the pinned `autogluon/mitra-regressor` snapshot and validates scikit-learn's diabetes table into an input manifest, with a recorded rejection probe. It checks cross-split overlap, registers Mitra in context and scores it on a holdout and an independent test next to Dummy mean/median, LightGBM and Random Forest baselines. It writes the evaluation report, exports the AutoGluon predictor bundle and reloads it through `safe_extract_archive` and `validate_artifact_directory` with an explicit tolerance. Its prose on point estimates, the absence of uncertainty intervals and the artifact's data obligations is accurate. The recorded Kaggle T4 run of this exact blob passed 9/9 code cells in one pass, with no restart. A CPU run in this review reproduced its metrics to two decimal places. The optional fine-tuning selection code, which crashes in the sibling classifier notebook, works here.

Two problems stand in the way of `Ready for intended use`:

1. **Mitra is not conditioned on the rows the notebook says it is (MRC-M1).** AutoGluon silently holds out 20% of the support rows, so Mitra's context is 212 of 265 rows. The baselines are fitted on all 265, yet the notebook calls this "the exact same support rows" and cites EVAL15 for it. The exported provenance records `train_rows_used: 265`.
2. **The guided layer is largely absent (MRC-M2).** The notebook declares `GUIDED`, but there is no how-to-use guidance, Input → Model → Output contract, expected-result notes, prediction prompts, checkpoints, glossary, troubleshooting or conclusion template. The 1,035-line carrier cell is not labelled or collapsed as infrastructure.

## 1. Review contract and evidence

| Item | Value |
|---|---|
| Declared profile / mode | `E2E` / `GUIDED` (`metadata.dimer`, opening cell) |
| Declared spec | DIMER Notebook Specification **2.0**, standalone (§4) |
| Spec baseline applied | NOTEBOOK_SPEC **2.2**. ID citations below are 2.2 IDs |
| Generator | `tools/build_notebook.py` (build_notebook.py/2) with template `tools/notebook_template.py`. Carried module `mitra_pipeline/tutorial_api.py` at `b249861b89bd`, sha256 `ad5f2f3a…` |
| Intended audience | Not stated explicitly. Prerequisites say "basic pandas" plus holdout, independent test, MAE, RMSE and R² |
| Supported runtime | Colab or Jupyter, Python 3.10–3.13. Default path on CPU, CUDA used if present. Fine-tuning needs a GPU |
| Promised outcomes | Verified acquisition; validated support data with overlap and cap reports; in-context evaluation on holdout and independent test against training-mean, median, LightGBM and Random Forest baselines; optional GPU fine-tuning with holdout-only selection; evaluation report; optional new-data inference; predictor bundle export and fresh reload; BYOD through the same cells |
| Cells | 21 cells: 9 code, 12 markdown. Code cells 3, 5 (1,035-line carrier), 7, 9, 11, 13, 15, 17 and 19 |

**Existing execution evidence.**

- `docs/release-verification.md` records a Kaggle T4 run of commit `7a4efc7`, blob `d32d6f6b45c6`, as **PASSED** (9/9 cells, 159.4 s). That is the reviewed blob; the notebook has not changed since.
- The executed notebook and `run_summary.json` are in `.agent/backups/kaggle-pass-2026-09-14/out/dimer-nb2-mitra-regressor/v2/evidence/`. They show:
  - one pass, with no restart, and a clean Hugging Face cache;
  - the Kaggle image's torch 2.10.0 replaced by AutoGluon's torch 2.9.1, with nothing imported before the install cell, and pip's dependency-resolver `ERROR` lines in the cell-3 output;
  - two Tesla T4 GPUs, device `cuda`.
- That run used an nbclient executor with a shim cell, not a browser `Run all` on Colab.
- No Colab execution of any revision is recorded. No BYOD, fine-tuning or new-data upload run is recorded.

**Journeys and evidence basis**

| Journey | Basis | What was done |
|---|---|---|
| First-time learner | Source inspection | All 21 cells read in order against GDL/UX/EVAL/DAT |
| Clean default | Documented execution (Kaggle T4, exact blob) **and** direct execution (CPU, see deviations) | Code cells 3–19 ran in order in one namespace, with the install skipped; every cell passed. Holdout and test metrics equal the Kaggle record to two decimal places (Mitra holdout MAE 40.140 vs 40.133) |
| Active learning | Direct execution (CPU) | (a) `EVAL_METRIC` changed to `root_mean_squared_error`, cells 13–19 rerun: all pass, AutoGluon and the run metadata pick up the change, every metric is unchanged (MRC-m4). (b) Cell 13 with `RUN_FINE_TUNING=True`, the GPU check stubbed and the candidate's fit replaced by a **stand-in** that reuses the pretrained predictor (+5 or 0 shift): selection runs and keeps the pretrained model, `selection_basis` `holdout:mean_absolute_error`. (c) Real CPU gate: GPU refusal message, after a 14 s pretrained refit |
| Reuse and recovery | Direct execution (CPU, `google.colab.files.upload` stubbed) | Insurance pre-split CSVs with `TARGET_COLUMN='charges'` through cells 9–19, plus 20-row `NEW_DATA_PATH` inference: all pass. Ames single-CSV upload (`SalePrice`) through cells 9–19: all pass, no test partition. Six invalid inputs (MRC-m3). Colab upload dialog not verified |

**Deviations and limitations.**

- **Host and environment.** This review ran on a Windows CPU host with about 12 GB of free RAM, using `mitra-regressor-pipeline/.venv` read-only. That environment has the notebook's exact pins: autogluon.tabular 1.5.0, lightgbm 4.6.0 and huggingface-hub 0.36.2, with torch 2.9.1+cpu. Its scikit-learn is 1.7.2; the Kaggle run had 1.6.1.
- **No memory override.** `MAX_MEMORY_USAGE_RATIO` stayed at the notebook default of 1.10 in every run.
- **Weights.** The Mitra snapshot was copied in with its digests checked, so the Hub download was not exercised here. The Kaggle record exercises it.
- **Install.** Cell 3 ran with `DIMER_NOTEBOOK_CI_PREINSTALLED=1`, so the pinned install and its restart guard were not exercised here. The Kaggle record exercises them.
- **Not verified.** A real GPU fine-tune was not run. A Colab browser `Run all` was not run. Learner observation was not attempted.
- **Probe wall time.** About 6.5 minutes.

## 2. Separate judgments

- **Technical correctness:** The default path is sound and reproducible: hosted run, plus this review's CPU rerun with matching metrics. Validation, overlap reporting, alignment, archive safety, verify-before-deserialise and the reload tolerance all behaved as documented. The fine-tuning selection code (holdout-only, strict improvement, 50-row floor, test-only warning) works with a stand-in candidate. One defect: the support-row count misreports what Mitra was conditioned on (MRC-M1).
- **Promise fulfilment:** Most promises are delivered and observable, and BYOD reaches every downstream stage for a representative pre-split table and a single CSV. One promise is not delivered as stated: baselines "fitted on the exact same support rows" as Mitra (MRC-M1).
- **Learner experience:** The notebook is accurate but dense, and is written as an engineering record. The opening cell alone is 5,392 characters, and it addresses a reviewer about an "open decision" in the spec. The guided layer is absent (MRC-M2). The model comparison is printed without help reading small MAE differences (MRC-m2). Cells 3 and 13 show pip `ERROR` lines and about 85 lines of AutoGluon logging, unexplained (MRC-m5). The `EVAL_METRIC` control appears to do nothing on the default path, and nothing says why (MRC-m4).
- **Spec conformance (2.2):**
  - OUT8 (MUST): the adaptation provenance misstates the support rows used (MRC-M1).
  - DAT12 (MUST): the target rule and the row/feature limits are not stated before the learner selects or uploads data, although the opening says they are (MRC-m3).
  - SHOULD shortfalls: GDL1–GDL14, UX4, UX8 and UX10.
  - The notebook declares spec 2.0, and the repository documents disagree about status and spec version (MRC-m6).
  - RUN1/RUN10/ENV6 are met on the hosted record: one pass, no restart.
  - FT2/RUN7: in-context support registration is an adaptation stage under §FT, so the default path satisfies FT2. The opening's "open decision" sentence is unnecessary (MRC-m1).
  - EVAL14/ART8: selection metric, selection split and exported variant are recorded (direct execution with a stand-in candidate).

## 3. Findings

### MRC-M1 (Major): Mitra's in-context support is 80% of the stated support rows; the "same rows" comparison and the provenance are wrong

- **Cell/section:** cell 13 (Section 6) and the carried `fit_mitra_predictor` (`mitra_pipeline/tutorial_api.py`, `predictor.fit(...)` at line 273). Related places:
  - the Section 6 prose, "fitted on the exact same support rows … scored on the exact same partitions (EVAL15)" (template line 216);
  - the opening, "in-context conditioning on the training split";
  - the cell-19 `run_metadata['train_rows_used']` (template line 403).
- **Observed issue:** `TabularPredictor.fit` is called without `tuning_data`. AutoGluon therefore logs "Automatically generating train/validation split with holdout_frac=0.2, Train Rows: 212, Val Rows: 53", and its Mitra model keeps only the train part as its context.
  - The registered Mitra predictor is conditioned on **212** of the 265 support rows.
  - Dummy mean/median, LightGBM and Random Forest are fitted on all **265**.
  - The exported `tutorial_run_metadata.json` records `train_rows_used: 265`.
  - The log line `-48.1091 = Validation score (-mean_absolute_error)` is computed on the 53 hidden rows and is never explained. It sits next to the holdout MAE of 40.13, with the opposite sign.
  - On the Insurance BYOD table the context is 641 of 802 rows.
  - The prose cites "EVAL15" for equal conditions; in spec 2.2, EVAL15 is the single-metric rule.
- **Consequence:** The central comparison is presented as equal-condition when it is not. Here it disadvantages Mitra, so the notebook does not inflate it. The deployed bundle's provenance misstates what the model was conditioned on, and a learner who reads "-48.1" in the log cannot tell which split produced it or why it is negative.
- **Evidence:**
  - Documented execution (Kaggle T4 log, cell 13): "holdout_frac=0.2, Train Rows: 212, Val Rows: 53" and "-48.1091 = Validation score (-mean_absolute_error)".
  - Direct execution (CPU): the fitted model's `model.X` has 212 rows, `load_data_internal('val')` has 53 rows, `X_train_tree` has 265 rows and `run_metadata.train_rows_used` is 265 (`P1_default`). Insurance 641/802 (`P4_byod.R5_presplit_insurance_full`).
  - Source inspection: neither `tuning_data` nor `holdout_frac` nor `refit_full` appears in the notebook (`S_static`).
- **Recommended correction:** Condition Mitra on the full support set and state it. Options:
  - Pass AutoGluon a fixed, documented split: carve a separate validation slice from support and report it, or supply `tuning_data` only if that partition is not also used for selection.
  - Or call `predictor.refit_full()` and export the refit model, so the context equals the support rows.
  - Or fit the baselines on the same 212 rows that AutoGluon kept.

  In every case, print the context size and record it truthfully in `run_metadata`. Explain AutoGluon's internal validation score and its sign, or suppress it. Correct the EVAL15 citation.
- **Acceptance check:** After cell 13, the printed Mitra context size equals `len(X_train_tree)` and equals `run_metadata['train_rows_used']`. Alternatively, the notebook prints both numbers and the prose states the difference. The Section 6 prose no longer claims "exact same support rows" unless the first condition holds.
- **Spec:** OUT8, FT3, GDL7, UX2. Comparison validity is framework §2 dimension 3.

### MRC-M2 (Major): declared GUIDED, but the guided layer is absent and infrastructure dominates

- **Cell/section:** the whole notebook, especially cells 0–2, cell 5, and the markdown before cells 9–19. Generator: `tools/notebook_template.py` markdown blocks; `tools/build_notebook.py` carrier cell.
- **Observed issue:** Missing elements, with the spec ID each would satisfy:

  | Missing element | Spec ID |
  |---|---|
  | Stated audience | GDL1 |
  | How to use (Run all, forms, which cells are infrastructure) | GDL2 |
  | Roadmap | GDL3 |
  | Input → Model → Output contract | GDL4 |
  | Observable objectives (the objectives are one long sentence of procedures) | GDL5 |
  | Glossary (in-context conditioning, support rows, R², independent test) | GDL6 |
  | Question or prediction before the baseline comparison | GDL7 |
  | "Expected result / What to notice" notes | GDL8, UX4 |
  | Checkpoints with sample answers | GDL9 |
  | Predict → change → run → observe → explain activity (the "Next experiments" are a list of switches) | GDL10 |
  | Section syntheses | UX8 |
  | Troubleshooting (Hub download, memory skip and `MAX_MEMORY_USAGE_RATIO`, GPU, BYOD target name) | GDL13 |
  | Conclusion template | GDL14 |

  The 1,035-line carrier (cell 5, 44,899 characters) has no Infrastructure label or "you may run without reading" note, and no `cellView: form` / hidden-source metadata (GDL11, GDL12). None of the guided markers appears in any markdown cell (`S_static.guided_markers_case_insensitive`).
- **Consequence:** A learner who meets the stated prerequisites can run the notebook but gets no help deciding what normal output looks like, or what the comparison means. They meet about 85 lines of AutoGluon log and a 45 kB carrier before the lesson.
- **Evidence:** Source inspection, cells 0–20; static marker scan (`S_static`).
- **Recommended correction:** Add the GDL elements in the template's markdown:
  - who the notebook is for and how to use it;
  - Input → Model → Output;
  - a prediction prompt before Section 6, for example "Will a pretrained in-context model beat Random Forest on 265 rows? By how many target units of MAE?";
  - "What to notice" after Sections 3, 5, 6, 7 and 9;
  - two checkpoints with collapsible answers;
  - one bounded predict–change–run–explain activity (for example `EVAL_METRIC` with the Insurance table, where MAE and RMSE rank the models differently);
  - a troubleshooting table that includes the memory skip and `MAX_MEMORY_USAGE_RATIO`;
  - a conclusion template.

  Label and collapse cell 5 as Infrastructure.
- **Acceptance check:** The rendered notebook contains each element named above. Cell 5 has `cellView: form` (or equivalent hidden-source metadata) and an Infrastructure label. A learner-facing troubleshooting entry names `MAX_MEMORY_USAGE_RATIO`.
- **Spec:** GDL1–GDL14 (SHOULD), UX4, UX8.

### MRC-m1 (Minor): the opening is a reviewer-facing wall of text

- **Cell/section:** cell 0 (5,392 characters in one markdown cell); template line 44.
- **Observed issue:** The "Run all" paragraph tells "a reviewer reading NOTEBOOK_SPEC 2.0 RUN7/FT2" to "treat that as an open decision". Capability, standalone carrier, Run all, BYOD, model notes, objectives and non-goals are packed into one cell.
- **Consequence:** A learner cannot tell what is essential, and the first instruction they read is addressed to someone else. The "open decision" is moot under spec 2.2: §FT names in-context support registration as an adaptation stage.
- **Evidence:** Source inspection; `S_static.reviewer_facing_open_decision_sentence` is true.
- **Recommended correction:** Remove the reviewer sentence. Split cell 0 into a short learner opening and a collapsible "About this notebook" block.
- **Acceptance check:** No learner-facing cell mentions reviewers or spec IDs as open decisions. Cell 0 is at most about 1,500 characters.
- **Spec:** GDL1, GDL2, GDL12.

### MRC-m2 (Minor): the model comparison is printed without help reading small differences

- **Cell/section:** cell 13 table; "Interpretation and limits" (template line 449).
- **Observed issue:** On the default sample, Mitra's holdout MAE is 40.1 against Random Forest's 42.3 and LightGBM's 44.9, on 88 rows. On the 89-row test partition, Mitra and Random Forest are 44.5 and 44.7, a gap of 0.2 target units. R² is 0.56 on holdout and 0.48 on test for Mitra. The interpretation then says "The executable baselines show when the foundation model adds value on this table", with no prompt to compare the gap with the spread between holdout and test, or to note that a single seeded split has no dispersion estimate. On the Insurance BYOD table, Random Forest has the lowest holdout MAE (2,540 vs Mitra 2,655) while Mitra has the lowest RMSE (4,371 vs 4,681–4,695), so the "winner" depends on the metric, and nothing helps the learner see this.
- **Consequence:** A learner is likely to read a 2-unit holdout MAE gap as Mitra being better, and to miss that the test result is a tie.
- **Evidence:** Documented execution (Kaggle table) and direct execution (`P1_default`, `P4_byod.R5_presplit_insurance_full`).
- **Recommended correction:** After the table, add a "What to notice" that compares the holdout gap with the holdout-to-test shift, points to RMSE and R² alongside MAE, and repeats that one seeded split carries no dispersion estimate. Optionally add a paired bootstrap interval, as the FreshRetailNet workshops do.
- **Acceptance check:** The markdown after cell 13 asks the learner to compare holdout and test gaps and at least two metrics. No sentence implies superiority from the single-split table alone.
- **Spec:** GDL7, GDL8, GDL14.

### MRC-m3 (Minor; DAT12 is a MUST): BYOD instructions don't match the upload contract, and the data limits come after the upload

- **Cell/section:** cell 0 BYOD paragraph; cell 1 Prerequisites; cell 8–9 (Section 4); "Next experiments" (template line 464).
- **Observed issue:**
  - "Next experiments" says to use the repository's `examples/sample-data` **archives** with `Upload pre-split train/val/test`. That branch accepts only `train.csv`, `val.csv` and `test.csv`. Uploading the ZIP fails with `Upload train.csv, val.csv, and test.csv together. Missing: ['test.csv', 'train.csv', 'val.csv']`, with no "extract the archive" hint.
  - Two of the three archives have targets `charges` and `SalePrice`, but `TARGET_COLUMN` stays `'target'`, and no BYOD text says to change it. The failure appears only in Section 5, after the upload: `pre-split upload:train: target 'target' not found.`, without naming `TARGET_COLUMN` or the available columns.
  - The opening says "Expected schema, ceilings and privacy guidance are stated in the Prerequisites and in Section 4". They are not: the numeric-target rule, `MIN_TRAIN_ROWS` (50), `MAX_TRAIN_ROWS` (10,000) and `MAX_FEATURES` (500) are first printed in Section 5, after the upload (`S_static.pre_upload_markdown_states`).
  - `VALIDATION_SPLIT` outside 0.05–0.40 is rejected only after the upload dialog has been answered.
  - Selecting an upload source without `USE_BYOD` raises "Set USE_BYOD=True …", which is actionable, but the two controls are redundant.
- **Consequence:** A learner following the notebook's own next step fails twice before reaching a working BYOD run, and cannot check a table against the limits before uploading it.
- **Evidence:** Direct execution, CPU, upload stubbed (`P4_byod` R1, R2, R3, R4b). Positive paths: Insurance with `TARGET_COLUMN='charges'` reached every stage (validation, overlap report, Mitra, baselines, report, export, verified reload, 20-row `NEW_DATA_PATH` inference); Ames single CSV with `SalePrice` reached every stage with no test partition. Invalid inputs were refused clearly: a text target (`target must be numeric; invalid examples: ['low', …]`) and an inference CSV missing `age`.
- **Recommended correction:** Say "extract the archive and upload its three CSVs", and name each archive's target column. State the target rule and the three limits in the Prerequisites or Section 4 markdown. Make the missing-target error say "set `TARGET_COLUMN` in Section 4" and list the columns. Check `TARGET_COLUMN` and `VALIDATION_SPLIT` in cell 9, before the upload is consumed. Either derive `USE_BYOD` from `DATA_SOURCE`, or document why both exist.
- **Acceptance check:** The markdown before cell 9 states the numeric-target rule and the values 50, 10,000 and 500. Uploading the Insurance ZIP produces a message that names extraction. Pre-split Insurance with the default `TARGET_COLUMN` fails in cell 9 with a message naming `TARGET_COLUMN` and the available columns.
- **Spec:** DAT12 (MUST), DAT19, UX10.

### MRC-m4 (Minor): the `EVAL_METRIC` control and the "Next experiments" give no expected outcome

- **Cell/section:** cell 13 form; cell 20 "Next experiments".
- **Observed issue:**
  - Changing `EVAL_METRIC` to `root_mean_squared_error` and rerunning Section 6 onward changes AutoGluon's metric and the recorded `eval_metric`, but every number in the table is identical: without fine-tuning, the metric only decides AutoGluon's internal validation score and the fine-tune selection. Nothing tells the learner this.
  - The "Next experiments" do not say which cells to rerun or what to look for.
  - When the fine-tuned candidate is not selected, it is printed as a dictionary only and never appears in the comparison table, so "watch the holdout-based selection" has little to watch.
  - On CPU, `RUN_FINE_TUNING=True` refits the pretrained predictor (14 s here) before refusing with the GPU message. The refusal itself is clear.
- **Consequence:** A learner who changes the one metric control sees no effect and cannot tell whether the control works.
- **Evidence:** Direct execution: `P3_active_eval_metric` (all metrics unchanged, `autogluon_eval_metric` and `run_metadata_eval_metric` both `root_mean_squared_error`); `P2_finetune_selection_standin`; `P2b_finetune_gate_cpu`.
- **Recommended correction:** Say next to `EVAL_METRIC` that it affects only the fine-tune selection and AutoGluon's internal score. For each experiment, state the cells to rerun and the expected observation. Add the candidate's holdout and test rows to the table. Move the GPU check before the pretrained refit.
- **Acceptance check:** The `EVAL_METRIC` markdown states its scope. Each "Next experiment" names its rerun range and what to look for. With `RUN_FINE_TUNING=True`, the table contains both Mitra variants.
- **Spec:** GDL10, UX5, UX4.

### MRC-m5 (Minor): unexplained errors, warnings and log volume in Sections 1 and 6

- **Cell/section:** cell 3 (install) and cell 13.
- **Observed issue:**
  - The Kaggle record's cell-3 output contains `ERROR: pip's dependency resolver …`, including "torchaudio 2.10.0+cu128 requires torch==2.10.0, but you have torch 2.9.1". The pinned install replaces the runtime's torch in the kernel. The run continued correctly, but nothing tells the learner the `ERROR` is expected.
  - Cell 13 prints AutoGluon's system-info block, a presets advertisement ("presets='extreme' … Massively better than 'best'"), "This metric's sign has been flipped", and about 85 lines of log in total. The CPU run adds einx deprecation and `pin_memory` warnings.
- **Consequence:** The learner can mistake the pip `ERROR` for a failed install, the advert for advice, and the negative validation score for a bug.
- **Evidence:** Documented execution: Kaggle T4 cell-3 stderr and cell-13 stderr. Direct execution: `P1_default.cells[5]` (85 output lines, warnings listed).
- **Recommended correction:** Fit with `verbosity=1` (or capture the log into a collapsed output). Explain the resolver lines in the Section 1 markdown, or move the workload into the uv isolated environment used fleet-wide (reference: `ast-audio-classification-pipeline/tutorials/DIMER_Sound_Event_Classification_Workshop.ipynb`), which leaves the kernel's torch untouched.
- **Acceptance check:** Default cell-13 output is under about 30 lines. Either cell 3's output has no `ERROR` line, or the Section 1 markdown names it as expected.
- **Spec:** UX4, GDL8, UX11.

### MRC-m6 (Minor): repository status records contradict each other and the notebook

- **Cell/section:** `docs/release-verification.md` header, "Automatic coverage" and "Current status"; `STATUS.md`; `README.md`; `tutorials/README.md`.
- **Observed issue:**
  - `release-verification.md` records the Kaggle T4 PASS of this blob, but its "Current status" says "No clean-runtime execution of the standalone notebooks has been recorded yet" and, in the same sentence, that GPU evidence "is now recorded below". Its header and static-coverage text still describe spec 1.1 metadata.
  - `STATUS.md` says "awaiting clean-runtime execution" and cites spec 1.1. `README.md` says spec 1.1 and "the standalone pair has not been executed yet". The notebook and `tutorials/README.md` say 2.0.
  - `tutorials/README.md` marks the run-all evidence "verified" while the status column says it "must be reviewed". Its "Data and evaluation" section describes a `SAMPLE_REVISION`, a `SAMPLE_CONFIGS` registry and a default FreshRetailNet path with 512 training rows; the E2E notebook has none of these (`S_static`) and defaults to the diabetes table.
- **Consequence:** A maintainer cannot tell whether the default path is evidenced, against which spec, or which sample it uses. A reader may rerun work, or promote on the wrong basis.
- **Evidence:** Source inspection of the four files at the reviewed commit.
- **Recommended correction:** Rewrite "Current status" to cite the Kaggle row and its boundary: nbclient with a shim, not a browser Colab `Run all`; default path only. Align `STATUS.md`, `README.md` and the release-verification header with the notebook's declared spec. Rewrite or scope the `tutorials/README.md` "Data and evaluation" section to the diabetes default and the upload path.
- **Acceptance check:** The four files name the same spec version, the same evidence state and the same default sample. "Current status" no longer contradicts the table.
- **Spec:** REL1, REL10.

### Suggestions

- **MRC-S1:** Regenerate against NOTEBOOK_SPEC 2.2. The notebook declares 2.0.
- **MRC-S2:** Use the Insurance BYOD result as a short interpretation activity on metric choice: Random Forest wins on MAE, Mitra on RMSE and R².
- **MRC-S3:** Add a predicted-versus-actual or residual plot for the holdout, so the learner can see where the errors are, not only their averages (UX11).

## 4. Readiness

**Needs revision.** No blocker. Two majors are open:

- MRC-M1: the comparison conditions and provenance are misstated;
- MRC-M2: the guided layer is missing for a `GUIDED` notebook.

OUT8 (MRC-M1) and DAT12 (MRC-m3) are MUST-level conformance gaps.

Remaining gates after the fixes:

1. An exact-revision clean `Run all` on Colab. Only a Kaggle nbclient run exists.
2. A real GPU fine-tuning run that reaches selection, export and reload.
3. A representative BYOD run under the hosted runtime (REL12).
4. Learner observation, if a claim about learning effectiveness is wanted.

## 5. Verified versus inferred

- **Verified by direct execution (CPU, labelled deviations):**
  - The default path passes, and its metrics equal the hosted record to two decimal places.
  - Mitra's context is 212/265 rows (641/802 on Insurance).
  - `run_metadata.train_rows_used` is 265.
  - The fine-tune selection code runs with a stand-in candidate and keeps the pretrained model when the candidate is not strictly better.
  - Changing `EVAL_METRIC` leaves every reported metric unchanged on the default path.
  - Insurance pre-split and Ames single-CSV BYOD reach every stage, including verified reload and `NEW_DATA_PATH` inference.
  - Six invalid inputs give the messages quoted.
- **Verified from documented execution:** the Kaggle T4 run of this blob, with one pass, no restart, 9/9 cells, and the pip and AutoGluon log lines quoted.
- **Inferred:**
  - That a real GPU fine-tune reaches the same selection code. The stand-in exercises the same lines after `candidate.fit`.
  - That the Colab browser `Run all` behaves like the Kaggle run, including the absence of a restart.
- **Most likely to be wrong:** MRC-M1's severity. The unequal conditions disadvantage Mitra, so the notebook does not overstate it, and AutoGluon's internal validation split is standard behaviour. It could be argued Minor. It is graded Major because two things are factually wrong: the stated equal-condition comparison (the central demonstration), and the exported provenance, which records 265 support rows for a model conditioned on 212.

Probe files: `mitra_regressor_colab_Review_Probes.zip` (`run_probes.py`, `results.json`, `source_manifest.json`).
