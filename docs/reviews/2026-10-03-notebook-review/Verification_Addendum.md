# Verification addendum: v1 regression workshop review (FRR1)

The review in this folder was produced on 2026-10-03 by a separate review pass. Before any fix, it was treated as a hypothesis and re-verified against the source.

## Revision check

- Reviewed commit `a9a6f056ebd52434cdcfd2b42b29bd9b83add453` equals `origin/main` and the GitHub API `main` at fix time.
- The notebook blob at that commit is `6508437e79c4db737d19dfa752de06d7e55cdbef`, as the review states.

## Majors

| Finding | Verdict | How it was re-verified |
|---|---|---|
| FRR1-M1 | Confirmed | Source: `SESSION=Session(SESSION_ROOT, …)` is built in 2.1, while 0.3 only replaces `SESSION_ROOT`, and `Session.assert_development` raises the frozen error. The cell 61 prose says "Leave **Freeze now** off" while `FREEZE_NOW = True`. Direct (CPU, revision 3.2.0): after a default Run all, 4.2 raises the frozen error; following only the new text (0.3 with a new experiment, then Run after from 1.1) completes 30/30 cells in a new experiment folder with `n_estimators=100` frozen |
| FRR1-M2 | Confirmed | Source: unguarded `next(...)` in 8.3. Reproduced with the review's own `cell_83_probe` on the reviewed notebook: `StopIteration` for `tabiclv2_icl,mitra_icl` |
| FRR1-M3 | Confirmed | Source: no guard on the R² denominator. Reproduced with `cell_83_probe`: target `[0, 1]`, 1,000 draws, 477 non-finite draws per model and both `ci_low` values NaN |
| FRR1-M4 | Confirmed | Source: `RUN_MITRA_FINETUNED = False`, `RUN_TABICLV2_FINETUNED = False`; the runner raises when fine-tuning runs without CUDA; research question 2 and checkpoint question 4 carry no optional label. Fixed by labelling it an optional GPU extension, not by dropping the objective |
| FRR1-M5 | Confirmed, one detail corrected | The README presents both notebooks without saying which to use and calls the guided edition "the same pipeline and defaults". The diabetes-sample sentence is in "Runtime and model contract" (it describes the generated notebooks), not in "Overview and workflow" as the review says; it was still clarified |

## Minors (all re-checked; at least three required)

- **FRR1-m1:** confirmed; `np.mean(gain)` in the reviewed 8.3 source.
- **FRR1-m2:** confirmed; 5.6 and 8.2 plot `.mae` with an MAE label, and the template compares validation MAE with test MAE.
- **FRR1-m3:** confirmed; 8.1 raises `ValueError("Freeze successful runs in 8.0 first.")`.
- **FRR1-m4:** confirmed from the pinned sample (SHA-256 `6534230e…`): zero-target rate is 3.37% train, 5.0% validation, 4.38% test and 3.94% overall; `DATASET_PROVENANCE` has no local-path key; quartile banding of validation `stockout_hours` gives 2 bands, because 59.5% are zero.
- **FRR1-m5:** confirmed; `_v2` workspace and owner names, an estimate with no number, and the 2.1 declaration.
- **FRR1-m6:** confirmed; no troubleshooting, sample answers or Input → Model → Output text in v1.
- **FRR1-m7:** confirmed; `make_result_record` is defined in 4.1 and never called, and the tables come from `workshop_core.regression_metrics`.

No finding was refuted. FRR1-S1 is out of scope and was not addressed.

## Evidence boundary

The re-verification used source inspection, the review's probes, and direct CPU execution on local Windows with the foundation models switched off. It is not Colab evidence and does not cover the foundation-model stages.
