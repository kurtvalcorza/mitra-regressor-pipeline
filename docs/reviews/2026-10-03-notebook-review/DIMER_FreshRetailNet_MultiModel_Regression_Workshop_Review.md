# Review: `DIMER_FreshRetailNet_MultiModel_Regression_Workshop.ipynb` (v1, revision 3.1.0)

**Framework:** Notebook Review Framework v1 (skill copy, sha256 `a7cc6611…`). **Requirements baseline:** NOTEBOOK_SPEC 2.2 (ml-worker `origin/main` `b1cfe133`). **Finding prefix:** `FRR1`. **Review date:** 2026-10-03. Review only; nothing in the repository was changed.

**Readiness: Needs revision.** No open Blocker. Five Major findings; the release gate (REL1) is also unmet because no execution evidence exists for this blob.

---

## 1. Scope and evidence

### Review contract

| Item | Value |
|---|---|
| Repository | `kurtvalcorza/mitra-regressor-pipeline` |
| Notebook | `tutorials/DIMER_FreshRetailNet_MultiModel_Regression_Workshop.ipynb` (the v1, not `_v2`) |
| Revision | `main` = `a9a6f056ebd52434cdcfd2b42b29bd9b83add453` (GitHub API, 2026-10-03); notebook git blob `6508437e79c4db737d19dfa752de06d7e55cdbef` |
| Notebook revision | `metadata.workshop_revision` 3.1.0; `metadata.dimer.clean_runtime_evidence: pending` |
| Declared spec | 2.1 (cells 0, 1, export manifest). The current baseline is 2.2 |
| Profile / mode | `E2E` / `WORKSHOP` |
| Audience / prerequisites | "College students with introductory Python and machine-learning experience" (cell 0) |
| Supported runtime | Google Colab, GPU recommended (`metadata.accelerator: GPU`); isolated `uv` Python 3.12 environments per foundation model; CUDA required only for optional fine-tuning |
| Promised outcomes | Cell 0, "What you will do" 1–8; the main research question plus 3 secondary questions (cell 2); 9 learning objectives (cell 3) |
| Shape | 77 cells (33 code, 44 markdown); 0 saved outputs; nbformat 4.5; all 33 code cells compile |

### Evidence used

| Basis | What |
|---|---|
| Source inspection | All 77 cells, including the embedded `CORE_SOURCE`/`EXECUTION_SOURCE`; `tutorials/README.md`; `docs/release-verification.md`; the 2026-09-27 v2 review |
| Documented execution evidence | **None for blob `6508437e`.** `docs/execution-evidence/` holds only `_v2` runs. Corroborating but **not counted**: 32 of 33 v1 code cells are byte-identical to code cells of the 2026-09-26 executed `_v2` evidence (commit `087b2fb`, 36/36 cells executed, 0 errors), in the same order. The one difference is cell 1.1, whose BYOD branch differs; the default download branch is the same. The 3 extra `_v2` cells only read state |
| Direct execution | `run_probes.py` on a **local Windows CPU** (Python 3.12.10, NumPy 2.5.2, pandas 3.0.5, scikit-learn 1.9.0, LightGBM 4.6.0). **Not Colab.** Stand-ins: every foundation-model condition is switched off, so no uv environment, checkpoint or runner is executed. Cell 0.3's `/content` root is redirected to scratch. Form fields are overridden in memory, as an executor would. Section 1.1 downloads the pinned sample for real |
| Learner observation | None |

### Journeys

| Journey | Basis | Result |
|---|---|---|
| First-time learner | Source inspection | **Needs revision.** The default Run all freezes the experiment, which makes the forms' invitations to experiment a dead end (FRR1-M1). The fine-tuning question cannot be answered on the default path (FRR1-M4). There are no troubleshooting notes, sample answers or labels on infrastructure cells (FRR1-m6) |
| Clean default | Direct execution, CPU stand-in; Colab **not verified** | The classical-only Run all completed **33/33 cells** and printed "Run-all complete: validation, freeze, artifact reload, independent test evaluation, inference preview, and export succeeded". The pinned download verified SHA-256 `6534230e…`. 7 runs were frozen (5 baselines + 2 LightGBM ablations); 7 test rows were scored and exported. Foundation-model stages were not executed. The default Colab path for this blob remains unverified |
| Active learning | Direct execution | **Before the freeze** (as the 8.0 prose asks): changing `RF_TREES` 300→100 propagated correctly. The frozen parameters were `n_estimators=100`, the test RF MAE was 0.4406 versus 0.4372 at 300 trees, and nothing was stale. **After the default Run all:** see FRR1-M1 |
| Reuse and recovery | Direct execution | BYOD path mode with a nested ZIP repacked from the sample completed 33/33 (classical only). Three invalid inputs were rejected before any fitting with messages that name the condition: missing `lag_1`, a non-ZIP file, and a missing path. A foundation-only selection fails in 8.3 (FRR1-M2). R² with a tiny test set gives NaN intervals (FRR1-M3) |

**Limitations.** No GPU. No foundation model, uv environment, checkpoint download, fine-tuning or Colab UI was exercised. The upload-dialog BYOD branch was not exercised. Install time and storage were not measured.

---

## 2. Separate judgments

- **Technical correctness:** the default classical path, the freeze/no-refit test, the bootstrap and the export work as coded (direct, CPU). The provenance and receipt machinery is careful. The defects sit in the transitions: the post-freeze recovery path (M1) and the 8.3 edge cases (M2, M3, m1).
- **Promise fulfilment:** the main question can be answered on the default path. Secondary question 2 (fine-tuning) cannot; fine-tuning is off by default and requires CUDA (M4). Secondary question 3 (stockouts) is served by the 6.1/6.2 ablations and the 7.1 bands, but the bands collapse to 2 (m4).
- **Learner experience:** the scientific framing is strong: the test-set rule, the baseline ladder, the warnings against overclaiming, censored demand and the bootstrap limitation. But the notebook gives no help with recovery or troubleshooting. It also teaches "leave Freeze off while exploring" while shipping it on.
- **Spec conformance (2.2):** see §5. REL1/REL10 are **unmet** for this blob. UX1 is partially met (fine-tuning objective). GDL1–15 (SHOULD) have several gaps. RUN7/FT2 are judged met through in-context conditioning and classical fitting; this is inferred, because in-context learning was not executed here.

---

## 3. Findings

### FRR1-M1 — Major: after the default Run all, the documented route back to experimenting is a dead end

**Cell/section:** 0.3 `START_NEW_EXPERIMENT`; 2.1 (`SESSION=Session(...)`); 8.0 `FREEZE_NOW = True`; the cell 61 prose.

**Observed:** `FREEZE_NOW` and `EVALUATE_FROZEN_TEST` default to `True`, so a default Run all freezes and tests every run. The cell 61 prose says "Leave **Freeze now** off while exploring". The forms throughout (`RF_TREES`, `ABLATION_MODEL`, `FOUNDATION_ABLATION_GROUP`, model toggles) invite changes. After Run all, any such change raises `This experiment is frozen. Start a new experiment before fitting or tuning again.` Doing exactly that (tick Start new experiment in 0.3, run it, rerun the changed cell) raises **the same error again**. Cell 0.3 creates a new `SESSION_ROOT`, but the `SESSION` object is built in 2.1 and still points at the frozen experiment. Recovery works only after also rerunning 1.1, 1.2 and 2.1, and no notebook text says so. The 2.1 error `Start a new experiment in Section 0.3` has the same gap.

**Consequence:** the active-learning journey a first-time learner reaches most naturally (run everything, then explore) loops on an error whose stated fix does not work. The prose and the defaults also contradict each other about when to freeze.

**Evidence:** direct execution, `journey_default_classical.post_run_active_learning`:
1. 4.2 → RuntimeError (frozen).
2. 0.3 with a new experiment → OK, new `SESSION_ROOT` `experiment-43766a54…`.
3. 4.2 → the **same** RuntimeError, while `SESSION.root` is still `experiment-db41dd68…`.
4. 1.1, 1.2, 2.1, then 4.2 → OK.

`_v2` added a troubleshooting row for this ("…then rerun from S…"); v1 has none.

**Correction:** state in 0.3, in the frozen-error message and in the 2.1 message exactly which cells to rerun after starting a new experiment (from 1.1 onward), or make 0.3 rebuild `SESSION` itself. Reconcile the cell 61 prose with the `FREEZE_NOW=True` default: either explain that Run all freezes everything as a reference run and that exploring needs a new experiment, or change the guidance.

**Acceptance check:** after a default Run all, change `RF_TREES`, then follow only the instructions printed in the error and in the 0.3/8.0 text. 4.2 must succeed in a new, unfrozen experiment without the learner rerunning any cell the text did not name. No markdown cell may tell the learner to leave Freeze off while the default is on, unless it also explains the default.

**Spec:** UX10 (SHOULD), GDL10 (SHOULD), UX7.

### FRR1-M2 — Major: a foundation-only frozen selection crashes the bootstrap with `StopIteration` (REG-01 applies to v1)

**Cell/section:** 8.0 `SELECTED_MODEL_KEYS`; 8.3 `reference_key=next(key for key in classical_ranking if key in bootstrap_predictions)`.

**Observed:** the v1 source still has the unguarded `next(...)` that `_v2` 3.1.1 fixed. 8.0 accepts any successful keys, and the 8.2 title advertises "baseline-only and partial-model selections supported".

**Consequence:** a valid-looking selection (for example `tabiclv2_icl,mitra_icl`) fails after the expensive test stage with a bare `StopIteration` and no message. 8.4 and 10.1 are never reached in Run all.

**Evidence:** direct execution of v1's own 8.3 source with a crafted namespace. `cell_8_3_foundation_only` → `StopIteration`; the control with LightGBM present → OK.

**Correction:** port the 3.1.1 fix. Fix and record the comparator at freeze time. With no frozen baseline, report the per-model intervals and skip the paired comparison with an explanation.

**Acceptance check:** the four selections (all models, two foundation models only, one foundation model only, baselines only) each either complete 8.3–10.1 or are rejected in 8.0 with a message that names the comparator requirement. The comparator appears in the bootstrap table and the export.

**Spec:** UX10, EVAL14.

### FRR1-M3 — Major: R² bootstrap intervals become NaN on small test sets (REG-02 applies to v1)

**Cell/section:** 0.2 `PRIMARY_METRIC` (r2 allowed); 2.1 BYOD minimum of 2 rows per split; 8.3 `score_for` for r2.

**Observed:** `1-np.sum(error**2)/np.sum((truth-truth.mean())**2)` with no guard against a degenerate draw. This is the code `_v2` 3.1.1 replaced.

**Consequence:** on a small BYOD test set with R² selected, the notebook shows `ci_low`/`ci_high` as NaN with no explanation. A learner cannot tell whether the model or the method failed.

**Evidence:** direct execution of v1's 8.3 source with test target `[0,1]` and 1,000 draws (seed 0): **477/1000** draws non-finite, and both `ci_low` values NaN. This matches the v2 review's P3. On the pinned 1,600-row sample, R² as the primary metric completed and gave finite intervals (`journey_metric_r2`).

**Correction:** port the 3.1.1 handling. Mark R² intervals `unavailable` with a count of degenerate draws, refuse a constant test target with a pointer to MAE/RMSE, and warn at freeze time when R² is chosen with few test rows.

**Acceptance check:** the R² intervals for target `[0,1]` are shown as unavailable with a reason, no NaN reaches the table or plot, and the point R² of 0.96 is kept. A constant test target gives an actionable error.

**Spec:** EVAL3, EVAL5, UX10.

### FRR1-M4 — Major: the fine-tuning question and objective cannot be answered on the default path, and the notebook never says so

**Cell/section:** cell 2, secondary question 2; cell 3, objective "distinguish **in-context learning** from **fine-tuning**"; 5.1 `RUN_MITRA_FINETUNED = False`, `RUN_TABICLV2_FINETUNED = False`; checkpoint cell 52, question 4; conclusion cell 70 §4.

**Observed:** the default run has zero fine-tuned conditions. Both fine-tuning conditions need CUDA: the core raises `Mitra fine-tuning needs CUDA`. No cell tells the learner that question 2 and checkpoint question 4 ("Did a fine-tuned condition actually update weights?") need these switches and a GPU. No cell gives the expected result or cost of enabling them.

**Consequence:** a learner who follows the default path is asked a research question and a checkpoint question they have no data for. The likely result is a guess, or a "fine-tuning does not help" conclusion drawn from in-context results alone.

**Evidence:** source inspection (`static.finetune_defaults`, `research_question_finetuning`). The fine-tuning code was not executed.

**Correction:** either run a bounded fine-tuning condition by default on the GPU runtime the notebook already recommends, or label question 2, the objective and checkpoint question 4 as optional extensions that require enabling 5.1's fine-tuning switches on a GPU runtime. Give the time cost, say what evidence of a weight update to look for, and say how to answer when the switches are off.

**Acceptance check:** on the default path, every research question and checkpoint question either has supporting output from that run, or is explicitly marked optional with the exact switch and runtime needed to answer it.

**Spec:** UX1 (MUST: objectives correspond to executed code); GDL7. RUN7/FT2 are judged met through in-context conditioning and classical fitting, which §13 counts as adaptation.

### FRR1-M5 — Major: the tutorials README does not tell a learner which notebook to use, and calls v1 equivalent to the fixed v2

**Cell/section:** `tutorials/README.md`, the registry rows for v1 and `_v2` (both have Colab badges) and the "Guided edition" paragraph.

**Observed:** the README presents both notebooks with launch badges. It describes `_v2` as "the same pipeline and defaults" plus teaching material, and as "written for readers who are new to machine learning". It never says which notebook a learner or instructor should open, or that v1 (3.1.0) lacks the 3.1.1 fixes. On this commit, 4 of v1's 33 code cells differ from the current `_v2` (29/33 identical): cell 1.1 BYOD mode, 8.0 comparator, 8.3 bootstrap, and 10.1 export. Three Major findings fixed in `_v2` (REG-01, REG-02, REG-05) remain open in v1 (M2, M3, m1). The README's own "Overview and workflow" also mixes in diabetes-sample text ("the default sample is scikit-learn's bundled diabetes table") that belongs to another notebook.

**Consequence:** an instructor who picks the shorter v1 badge gets a notebook with known defects that the README implies are fixed. The audience distinction (beginner versus ML-literate) is implicit.

**Evidence:** source inspection of README; code-cell comparison (direct, `source_manifest.json`).

**Correction:** add a "Which notebook should I use?" note. Name the audience for each notebook. State whether v1 is maintained, superseded or frozen, and list the 3.1.1 fixes it lacks, or port them. Remove the "same pipeline" claim unless the code cells match.

**Acceptance check:** a reader of `tutorials/README.md` alone can say which notebook to open for (a) a facilitated session for ML-literate students and (b) self-paced beginners, and whether v1 carries the REG-01/02/05 fixes. Every equivalence claim between the two notebooks is true by code-cell comparison at the stated revisions.

**Spec:** SRC10, §27 registry.

### FRR1-m1 — Minor: `gain_vs_reference` is a bootstrap mean, not the observed difference (REG-05 applies to v1)

**Cell 8.3.** The table pairs full-test point scores with `np.mean(gain)` over draws.

**Evidence:** direct execution (default journey, reference `lightgbm`). The reported gains differ from the observed point differences by up to **0.0017 MAE**; for example, Random Forest is reported as 0.02312 against an observed 0.02290.

**Correction:** report the observed difference; use the draws only for its interval.

**Acceptance check:** for MAE, RMSE and R², every `gain_vs_reference` equals the difference of the displayed point scores in the favourable direction.

### FRR1-m2 — Minor: the conclusion template and plots stay MAE-centred and mix partitions

**Cells 5.6, 8.2, 70.** Both plots always chart MAE, with the label "MAE (normalized observed-sales scale)", even when `PRIMARY_METRIC` is rmse or r2. Template §2 hard-codes "an MAE of". Template §5 compares the stockout ablation on **validation** MAE but the month/weather ablation on **test** MAE.

**Evidence:** source inspection and direct execution (`journey_metric_r2`: R² was ranked while MAE was plotted).

**Correction:** drive the plot metric and the template wording from `PRIMARY_METRIC`, and use one partition for both ablation statements, stating which one it is.

**Acceptance check:** with `PRIMARY_METRIC="rmse"`, the 5.6/8.2 plots and the template refer to RMSE, and both ablation sentences name the same partition.

### FRR1-m3 — Minor: following the "explore first" instruction makes Run all stop at 8.1 with an error

**Cells 8.0–8.4, summary.** With `FREEZE_NOW=False`, 8.1 raises `ValueError: Freeze successful runs in 8.0 first.`, which stops Run all. With `EVALUATE_FROZEN_TEST=False`, 8.3 and 8.4 raise `RuntimeError`. The completion summary's "exploratory run" branch therefore cannot be reached.

**Evidence:** direct execution (`journey_explore_before_freeze.run_all_with_freeze_off_first_error`).

**Correction:** when not frozen, have 8.1–8.4 and 10.1 print a skip notice and continue, so the exploratory summary is reachable. Alternatively, tell learners to run up to 7.1 only.

**Acceptance check:** Run all with `FREEZE_NOW=False` ends at the summary cell and prints the exploratory-run message, with no exception.

### FRR1-m4 — Minor: fixed "expected output" text does not match what the run shows

- Cell 35 says "Approximately 3.4% of target observations are zero-valued". This is the training split only (3.37%); validation is 5.0%, test 4.4%, and all splits together 3.94%. The text stays fixed under BYOD.
- Cell 15 "Expected result" lists "the local archive path", but `DATASET_PROVENANCE` has no such field.
- Cell 7.1 bins validation `stockout_hours` with `qcut(q=4)`. 59.5% of the values are 0, so only **2** bands appear: `(-0.001, 5.0]` and `(5.0, 16.0]`. The notebook does not explain this, and zero-stockout rows are merged with rows of 1–5 hours.

**Evidence:** direct execution (`data_facts`); source inspection.

**Correction:** compute the zero rate in the cell output rather than in the prose, or name the split. Show the archive path or drop it from the expected list. Bin stockouts explicitly (0, 1–5, >5 hours) or explain the merging.

**Acceptance check:** every number and output item named in markdown appears in, or is computed by, the adjacent output for the pinned sample. The stockout analysis shows a separate zero-stockout group.

### FRR1-m5 — Minor: identity and naming drift

- The notebook declares spec 2.1, but the baseline is 2.2.
- The title is "DIMER Workshop: …" (GDL15 prefers "DIMER Notebook: …" for the file).
- "Estimated time" gives no estimate.
- v1 writes to `/content/dimer_tabular_workshop_v2`, claims owner `dimer-freshretailnet-workshop-v2` and records `adapter_version: classical-workshop-v2`. Cell 5 shows this `_v2` path to v1 learners, while a separate `_v2` notebook exists.

**Evidence:** source inspection (`static.workspace_v2_naming`, `estimated_time_text`).

**Correction:** re-declare against 2.2 or record why 2.1 is kept. Rename the title. Give a measured or labelled estimate (UX12). Use workspace and owner names that identify v1, or explain that both notebooks share the workspace.

**Acceptance check:** the notebook text contains no `_v2` identifiers unless they are explained, and the opening states a time estimate labelled as an estimate.

### FRR1-m6 — Minor: guided-layer gaps relative to GDL (all SHOULD)

The notebook has none of the following: a How-to-use section (GDL2), an Input → Model → Output contract (GDL4), sample answers for the six checkpoints (GDL9), a Predict → Change → Run → Observe → Explain activity (GDL10), or a troubleshooting section (GDL13). Infrastructure cells carry about 30 KB (0.3), 157 KB (5.2) and 22 KB (5.3) of embedded source without an **Infrastructure** label or collapse (GDL11). `_v2` supplies most of these.

**Evidence:** source inspection (`static.has_troubleshooting/has_sample_answers/has_input_model_output` all false).

**Correction:** add the missing layers, or state in the opening and the README that v1 is the facilitator edition for ML-literate students and point beginners to `_v2`.

**Acceptance check:** each of GDL2, GDL4, GDL9, GDL10, GDL11 and GDL13 is either present or explicitly waived in the opening, with a pointer to `_v2`.

### FRR1-m7 — Minor: Section 4.1 shows metric code that does not produce the tables

**Cell 4.1** defines `regression_metrics` and `make_result_record`. Every table is computed by `workshop_core.regression_metrics` through `load_results`. `make_result_record` is never called (`static.make_result_record_used_elsewhere=false`). The two implementations differ in their handling of R² for constant targets.

**Correction:** make 4.1 the implementation the pipeline calls, or label it as an illustrative copy and remove the unused helper.

**Acceptance check:** the metric function shown to learners is the one that produces the validation and test tables, or it is labelled as an illustration.

### FRR1-S1 — Suggestion: add a reference-run note

Saved outputs are cleared, which is correct (SRC4). A short "what a normal run looks like" note (shape of the tables, approximate MAE scale around 0.4 on the sample, runtime per model on T4), labelled as one reference configuration, would help learners and facilitators judge whether their run is normal. The 2026-09-27 `_v2` evidence could serve as the source.

---

## 4. Promise and objective tracing (summary)

| Claim / objective | Cell | Observable result | Verdict |
|---|---|---|---|
| Acquire and verify the pinned sample | 1.1–1.2 | SHA-256 `6534230e…` verified; 3 CSVs staged | Delivered (direct) |
| Validate data | 2.1–2.2 | Schema, target, row counts; BYOD rejections named | Delivered (direct) |
| EDA, test targets sealed | 3.1–3.6 | Train and val only; test rejected unless opted in | Delivered (direct) |
| Classical baseline ladder | 4.2–4.4 | 5 baselines; validation MAE 0.373–0.696 | Delivered (direct) |
| Run 4 foundation models locally | 5.1–5.6 | — | **Not verified** (not executed; different-blob evidence only) |
| Compare under one protocol | 5.6, 8.2 | Combined tables | Delivered for classical; FM not verified |
| Fine-tuning versus in-context | 5.1 | No fine-tuned run by default | **Not delivered on default path** (M4) |
| Stockout effect on error | 6.1, 6.2, 7.1 | Ablation tables; 2-band split in 7.1 | Partly delivered (m4) |
| Freeze, then independent test | 8.0–8.2 | No refit; artifacts reloaded | Delivered (direct, classical) |
| Uncertainty | 8.3 | Bootstrap CIs | Delivered on default; fails on M2/M3 paths |
| Evidence-based conclusion | 70–71 | Template + checklist | Delivered; MAE-centred (m2) |
| Export bundle | 10.1 | Receipts, CSVs, figures, manifest, ZIP | Delivered (direct) |

---

## 5. Spec conformance (2.2), separate from severity

| Requirement | Status |
|---|---|
| RUN1/RUN2/RUN3/RUN5 | CPU stand-in Run all passed with no edits, no upload and an automatic sample. **Colab not verified** for this blob |
| RUN7/FT2 | Met via in-context conditioning and classical fitting (inferred; in-context learning not executed). Fine-tuning is optional by design; see M4 |
| UX1 (MUST) | **Partial**: the fine-tuning objective has no executed counterpart (M4) |
| UX10, EVAL14 | Not met on the M1/M2/M3 paths |
| DAT10–DAT19 | Local positive and 3 negative BYOD cases OK (classical). REL12 hosted BYOD **not verified** |
| UNC6 | Met (8.4 states point predictions, no intervals) |
| SRC1/SRC4/SRC9 | Met (nbformat 4.5, outputs cleared, all cells compile) |
| GDL1–GDL15 (SHOULD) | Gaps (m5, m6) |
| **REL1/REL10** | **Unmet**: no execution evidence for blob `6508437e`. The README agrees ("unverified for Revision 3.1.0") |

## 6. Readiness

**Needs revision.** Remaining gates:
1. Resolve M1–M5.
2. Run a fresh Colab default Run all of the revised blob, recorded in `docs/release-verification.md` (REL1/REL10).
3. Run a foundation-only selection, R² as primary metric, and hosted BYOD positive and negative cases (REL12).
4. Optionally run one fine-tuning condition on GPU if M4 is fixed by enabling it.

## 7. What was verified versus inferred

- **Verified (direct, local CPU):** classical default path 33/33; the post-freeze recovery loop; the 8.3 `StopIteration` and NaN-interval paths using v1's own cell source; the bootstrap-mean gain discrepancy; BYOD positive and 3 negative rejections; data facts (zero rates, months 4–5/5/6, 2-band stockout split).
- **Inferred:** that the foundation-model stages run on Colab, from 32/33 code cells being identical to the 2026-09-26 `_v2` evidence (a different blob, not counted).
- **Only the maintainer or a Colab run can confirm:** the foundation stages, timing, GPU fine-tuning, and the upload-dialog BYOD branch.
- **Most likely to be wrong:** the severity of **FRR1-M4**. If the maintainer treats fine-tuning as an explicitly optional GPU extension, a labelling fix makes it Minor. The UX1 reading depends on whether "distinguish in-context learning from fine-tuning" counts as answered by the glossary alone.
