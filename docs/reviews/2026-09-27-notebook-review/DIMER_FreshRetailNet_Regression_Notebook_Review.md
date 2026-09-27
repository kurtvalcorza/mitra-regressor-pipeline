# FreshRetailNet Multi-Model Regression Notebook — Review

**Review date:** 27 September 2026  
**Framework:** Notebook Review Framework v1  
**Verdict:** **Needs revision for the full documented interactive/BYOD experience.** No default-path blocker was established by this review.

## 1. Review contract and evidence

| Item | Reviewed identity |
|---|---|
| Repository | `kurtvalcorza/mitra-regressor-pipeline` |
| Notebook | `tutorials/DIMER_FreshRetailNet_MultiModel_Regression_Workshop_v2.ipynb` |
| Commit | `dd7c8ecd49ad1c7de4cae5875152a71c40f431d7` |
| Notebook Git blob | `026741c58e5a9557d14320f72239afc411f2351a` |
| Declared profile / mode | `E2E` / `WORKSHOP` |
| Declared notebook specification | 2.1; the supplied current specification is 2.2, dated 2026-09-26 |
| Intended audience | Learners who can run notebooks and read basic Python, but are new to machine learning |
| Documented primary metric | Mean absolute error (MAE); the settings also expose RMSE and R² |
| Default learning task | Predict one numeric value of observed sales seven days ahead; compare classical baselines and four in-context tabular foundation-model conditions |
| Documented runtime | Colab T4; isolated Python 3.12 model environments |

**Pinned source:** [Reviewed notebook](https://github.com/kurtvalcorza/mitra-regressor-pipeline/blob/dd7c8ecd49ad1c7de4cae5875152a71c40f431d7/tutorials/DIMER_FreshRetailNet_MultiModel_Regression_Workshop_v2.ipynb).

The sample is `freshretailnet-h7`, pinned to repository revision `78e12407044bbca6ed8edbb9754bb33bf09117ac`, archive SHA-256 `6534230e9eb6a2e212b741c4c17d897a57338323eb011f35ba2dc3fb28d8bb7b`.

### What the evidence does and does not establish

This review combines targeted source inspection, learner-facing content inspection, inspection of saved output tables, the repository's recorded execution evidence, and reduced offline reproductions of three specific code paths. The probes transcribe the relevant expressions and use synthetic fixtures. They do **not** execute the entire notebook, its `Session` implementation, model adapters, foundation-model weights, a GPU runtime, or Colab.

The probe environment was Python 3.13.5, NumPy 2.3.5, pandas 2.2.3, and scikit-learn 1.8.0. It is not the notebook's documented T4 environment. No representative learners were observed. UI rendering, accessibility, fresh-install timing, complete model behavior, and all security/artifact boundaries have not been independently certified.

### Existing Colab execution evidence must be credited

The [release-verification record at the reviewed commit](https://github.com/kurtvalcorza/mitra-regressor-pipeline/blob/dd7c8ecd49ad1c7de4cae5875152a71c40f431d7/docs/release-verification.md) contains a later **Maintainer-supplied Colab execution — 2026-09-26** entry. It records:

- reviewed source commit `087b2fbfbbded007c0fcf3902f167b33b7232da9`;
- supplied executed-file SHA-256 `ae31481fa1fa5d8d478591ab5a16b9397e7ddf7ebea23447de832220927950af`;
- 36 executed code cells, no saved error outputs, and terminal completion/exports;
- default controls and a source comparison matching the reviewed PR;
- inspection of the supplied execution artifact, not an independently repeated execution.

That entry expressly supersedes an earlier pending-rerun item for its source/configuration. It would be incorrect to ignore it and report that no successful Colab evidence exists. Full downstream BYOD and other untested optional paths remain separate gates. This review did not independently compare every executable cell at `087b2fbf` with the current `dd7c8ecd` revision, so it does not upgrade the earlier record into exact-current-revision execution proof.

## 2. Summary judgment

The notebook implements a substantial comparative learning workflow, not merely a sequence of model invocations. Its orientation, baseline comparison, prediction prompts, row-level error inspection, experiment controls, and interpretation sections provide a useful instructional structure. Saved outputs show the four default foundation-model conditions and subsequent test results; those are recorded notebook results, not newly reproduced model scores.

The confirmed weaknesses are concentrated in **the transition from the reference run to learner-controlled settings and user data**. Three allowed-looking configurations have problems: selecting only foundation models for final testing; selecting R² with a very small test partition; and converting a tied BYOD target into three demand bands without checking whether those bands actually exist.

These do not establish that the pinned default MAE run fails. They do establish that the broader user-facing experience needs revision.

## 3. Prioritized findings

### REG-01 — Major: a foundation-only frozen selection has no bootstrap reference

**Location:** Section 8.0, `Freeze selected successful local runs`; Section 8.3, `Bootstrap confidence intervals for frozen test scores`.

**Affected journey:** A learner runs the comparison, then chooses only successful foundation-model keys in `SELECTED_MODEL_KEYS` for final evaluation.

**Observed implementation:** The visible selection cell accepts current successful keys. The bootstrap then constructs its reference from the classical baseline ranking and requires one of those baseline keys to exist in the frozen prediction registry:

```python
reference_key = next(
    key for key in classical_ranking
    if key in bootstrap_predictions
)
```

There is no default value or missing-reference branch in that expression. A registry containing `tabiclv2_icl` and `mitra_icl`, but no classical baseline, yields `StopIteration`.

**Direct evidence:** Probe P1 reproduces the visible selection-membership check and reference-selection expression with synthetic records. The foundation-only selection passes the former and fails the latter. Probe P2 includes LightGBM in the registry and selects it successfully. The complete freeze/test machinery was not executed in these probes.

**Learner consequence:** A valid-looking final selection can lead to a low-level exception after the costly modeling and test stages, blocking uncertainty analysis and downstream completion. The failure does not explain the unstated comparator requirement.

**Correction:** Choose and record the comparator before freezing. Either require its inclusion with an actionable preflight message, or allow foundation-only evaluation and explicitly omit the paired comparison while retaining individual intervals. Do not silently choose a new comparator from test performance.

**Acceptance check:** Exercise default/all-model, two-foundation-only, one-foundation-only, and baseline-only selections. Each must either complete the supported downstream path or reject the unsupported selection before final evaluation. The selected comparator and selection basis must be explicit in outputs and export.

**Requirement relevance:** Notebook Review Framework promise fulfillment and interaction/recovery; current-spec UX10 (actionable failure messages) and EVAL14 (explicit selection metric/split). This is an optional-selection defect, not a demonstrated RUN1 failure for untouched defaults.

### REG-02 — Major: R² bootstrap draws can produce undefined scores on accepted small test sets

**Location:** Primary-metric setting; Section 2 BYOD minimum-row validation; Section 8.3 `score_for` and percentile calculations.

**Affected journey:** `PRIMARY_METRIC = "r2"` with a small BYOD test set. This is not the default MAE/sample path.

**Observed implementation:** The notebook exposes R² as a primary metric and its visible BYOD row check permits two rows per split. The bootstrap's R² implementation divides by the within-resample target sum of squares without checking for zero variance:

```python
1 - np.sum(error**2) / np.sum((truth - truth.mean())**2)
```

Sampling test rows with replacement can produce a constant-target draw even when the original test target is nonconstant. Non-finite scores then reach percentile and paired-difference calculations without a defined policy.

**Direct evidence:** P3 uses two distinct test targets `[0, 1]` and predictions `[0.1, 0.9]`. Full-test R² is a finite **0.96**, consistent with scikit-learn. With the notebook's seed-zero, 1,000-draw resampling pattern, **477 draws** have non-finite R²; the computed lower percentile is `NaN`. P4 confirms that the MAE and RMSE branches remain finite on the same draws. This is a scorer/resampling probe, not a complete small-BYOD model run.

**Learner consequence:** A selectable metric can yield an unusable uncertainty table even though the full test score is valid. A novice has not been given the conditions under which R² ceases to be defined on bootstrap draws.

**Correction:** Define the statistical contract for R² uncertainty. Detect degenerate target draws, report their count, and use an explicitly justified procedure—or mark R² intervals unavailable and retain the point score. Where small data makes meaningful inference untenable, explain that limitation before the expensive path. Merely substituting zeros/ones or silently discarding arbitrary draws is not a defensible general fix.

**Acceptance check:** Test constant targets, a two-row nonconstant target, heavily repeated targets, and an ordinary larger target. No non-finite value should reach plotting or a purported confidence interval without an explicit unavailable/invalid status and explanation. Preserve the seed and uncertainty-method metadata.

**Requirement relevance:** Framework scientific validity and interpretation; current-spec EVAL3/EVAL5 (metric and estimation semantics), VAL5 (target rules), and UX10. The issue is not that every two-row regression task must be forbidden; it is that this uncertainty procedure has additional conditions the notebook does not handle.

### REG-03 — Major: the three-band exercise can collapse into a one-class task on BYOD

**Location:** Section 9.1, `From numbers to demand bands` and `Score regression predictions as demand bands`.

**Affected journey:** User-supplied regression targets with substantial ties, followed by the promised regression-to-classification exercise.

**Observed implementation:** The cell computes training tertiles, maps targets and predictions through `np.digitize`, and calls `balanced_accuracy_score`. It asserts the expected `[0.5, 0.9]` thresholds only for the pinned sample. For BYOD it does not check distinct cut points or resulting class coverage.

**Direct evidence:** P5 uses 100 training targets: 70 zeros, 20 ones, and 10 twos. There are three distinct numeric values, but both tertiles equal zero. The cell's transformation maps test values `[0, 1, 2]` to `high, high, high`. Constant numeric predictions also map to `high`, producing balanced accuracy **1.0**. P6 confirms that with distinct boundaries and all three classes represented, an always-one-band predictor scores **1/3** as the lesson describes.

**Learner consequence:** A learner can obtain an apparently perfect result after the task has silently ceased to be three-band classification. The metric is behaving consistently with the observed one-class labels; the defect is the missing derived-task validation and the resulting mismatch with the lesson.

**Correction:** Validate strictly increasing, meaningful band boundaries; display the resulting train/validation/test class coverage; and explicitly handle degenerate bands before scoring. Require learner-approved domain thresholds, skip the comparison with an explanation, or deliberately redefine the task and labels. Do not silently move cut points using test data to manufacture coverage.

The historical companion-classifier comparison also needs a BYOD boundary: its published reference scores concern the pinned companion sample, not arbitrary user data. Add a visible BYOD-specific notice or disable that comparison until equivalent data, splits, and target definitions are established.

**Acceptance check:** Test the pinned sample, unique/continuous BYOD targets, heavily tied targets with three distinct numeric values, and a constant target. A one-class derived task must never appear as successful three-class performance. Do not imply that unrelated BYOD scores are directly comparable with the companion notebook's saved results.

**Requirement relevance:** Framework promise fulfillment, scientific validity, and learner interpretation; current-spec classification requirements in §21.1 (class count/coverage and output semantics), VAL2/VAL5, and GDL8. The pinned sample's band stage was not shown to have this problem.

### REG-04 — Minor: the worked answers overstate invariance and do not consistently adapt to changed settings

**Location:** Test checkpoint and final conclusion template.

The answer key says the classical results from its reference run “do not change between revisions.” Fixed example answers are legitimate instructional aids, but they are not invariants: the notebook exposes tree counts, metric choice, dataset choice, and other settings. The conclusion template also remains MAE-centered when the user chooses RMSE or R² as primary.

**Correction:** Label answers with the exact reference dataset/configuration, state that alternative settings/data can change them, and make the principal conclusion follow the selected metric and comparator. Keep the worked reference visible as an example, not as an expected winner or acceptance oracle.

**Acceptance check:** Change one baseline parameter and the primary metric. Prompts and the conclusion must still describe what was actually selected and measured, without directing the learner to reproduce the historical numbers.

### REG-05 — Minor: the reported gain is a bootstrap average rather than the observed score difference

**Location:** Section 8.3, `gain_vs_reference`.

The table reports the full-test point score in the metric column, but fills `gain_vs_reference` with `np.mean(gain)` over bootstrap draws. These quantities need not equal the difference between the displayed point scores. For nonlinear metrics, bootstrap averaging also changes the estimator rather than merely adding uncertainty around the observed contrast.

**Correction:** Report the observed full-test score difference, using the documented favorable direction. Use paired bootstrap draws for interval estimation. A bootstrap-mean gain can be retained only under a distinct label.

**Acceptance check:** Assert that the displayed observed gain equals the appropriate difference between displayed full-test point scores for MAE, RMSE, and R². Record the comparator in the exported table itself.

## 4. Learner experience and promise-to-evidence assessment

| Promise / dimension | Assessment from available evidence |
|---|---|
| Concrete regression task and audience | Strong orientation: input features, point prediction, seven-day horizon, and observed-sales limitation are made explicit. |
| Compare four foundation models with classical baselines | Implemented, with saved four-model validation results and repository-recorded default execution evidence. No new foundation-model run was performed here. |
| Technical experiment control | Visible code uses session/development assertions, deep-copied controls, source identity checks, receipts, and a separate freeze/test workflow. This is not the classifier notebook's simpler state management. Its complete implementation was not exhaustively tested in this review. |
| Explain metrics and outputs | The lesson connects MAE, RMSE, and R² with error behavior and period changes, and includes a one-row walkthrough. R² bootstrap and reference-answer boundaries need the fixes above. |
| Meaningful learner action | Predict/compare prompts and feature-group ablations provide actual experimental choices. The presence of those activities is established; learning effectiveness was not measured. |
| Inference | The test stage is described as reloading artifacts and predicting held-out rows; §8.4 explicitly previews existing prediction-only output. The preview is not being mislabeled here as a fresh independent inference call. |
| Reuse and recovery | BYOD and model-selection controls exist, but selected configurations expose REG-01 to REG-03. Full BYOD execution remains an outstanding repository gate. |
| Completion and transfer | The conclusion template, submission checklist, and export stage provide an identifiable endpoint. The template needs to follow the selected metric; the band-transfer stage needs derived-task validation. |

### What to retain

Retain the guided structure rather than replacing it with a shorter execution-only notebook. In particular, keep the distinction between observed sales and unconstrained demand, the baseline ladder, the connection between numeric errors and row-level examples, and the explicit split between concept, experimental practice, and infrastructure.

Do not mechanically transfer findings from the classification notebook. The regression notebook visibly uses a different experiment-control design; this review did not reproduce the classifier's staging, fine-tuned ablation dispatch, or live-configuration freeze defects. Likewise, the saved regression bootstrap names LightGBM as its reference, so its LightGBM-versus-ablated-LightGBM comparison is not automatically the classifier review's wrong-comparator defect.

### One useful teaching enhancement — not a release defect

Add a **training-median** constant baseline alongside the training mean. It would make the distinction between an absolute-error objective and a squared-error objective tangible. The existing fitted and rolling-history baselines mean this omission does not invalidate the current model comparison; it is a teaching opportunity, not a mandatory new capability.

## 5. Verification ledger

| Check | Evidence / outcome |
|---|---|
| Original/reference Colab execution | Credited from the later maintainer-supplied execution record at source `087b2fbf`; not independently repeated here. |
| Exact current notebook, fresh T4 Run all | Not executed by this reviewer. |
| Default four-model saved outputs | Inspected as recorded output, not treated as newly reproduced scores. |
| Foundation-only reference selection | Reduced local reproduction: `StopIteration`. |
| Reference present | Positive control: LightGBM selected. |
| R² bootstrap with a two-row nonconstant test target | Reduced local reproduction: 477/1,000 non-finite draws; non-finite interval endpoint. |
| MAE/RMSE scorer controls | Finite on the same resampling fixture. |
| Tied-target band conversion | Reduced local reproduction: duplicate thresholds, one observed class, balanced accuracy 1.0. |
| Ordinary three-class band conversion | Positive control: one-band predictor scores 1/3. |
| Full BYOD / optional fine-tuning / documented fast path | Not executed here. |
| Observed learner walkthrough / Colab visual audit | Not performed. |

## 6. Readiness and remediation order

**Technical correctness:** Targeted defects confirmed in optional configurations; no default-path blocker established.  
**Promise fulfillment:** The reference comparative workflow is substantially implemented. The broader selection/BYOD promises require revision.  
**Learner experience:** Substantial scaffolding, with consequential edge-case gaps rather than a general absence of explanation.  
**Specification conformance:** Targeted findings and relevant requirement mappings, not an exhaustive 2.1 or 2.2 conformance certificate.  
**Overall:** **Needs revision.**

Prioritize three changes: make comparator eligibility explicit before final evaluation; handle undefined R² uncertainty; and validate the derived three-band task. Then correct reference-answer wording and the gain statistic.

After implementation, record a fresh default T4 run against the changed notebook identity and exercise the documented fast path, foundation-only selection, positive/negative BYOD, R² edge cases, and tied-target band conversion. Preserve existing valid execution evidence rather than rewriting it as if it never existed.

A representative learner walkthrough should then test whether the participant can explain the input/output contract, select an appropriate metric, identify the comparator, perform one controlled change, recognize when a derived task is invalid, and retain the exported result. These are proposed acceptance activities, not activities already completed in this review.

## 7. Companion evidence files

`DIMER_FreshRetailNet_Regression_Review_Probes.zip` contains `review_probes.py`, `probe_results.json`, and a README documenting provenance and limitations. The probes are intentionally reduced reproductions, not a Colab executor or a substitute for release qualification.

**No repository changes were made.**
