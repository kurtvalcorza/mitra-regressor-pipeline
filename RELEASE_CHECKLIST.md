# Release checklist — Mitra tabular-regression pipeline (issue #1)

Production-enablement gate for the validator + finetuner pair. Repo-side contract items are
satisfied on `main` (evidence below); the final gate is an on-platform DIMER verification the
platform team owns. See [DEPLOYMENT.md](DEPLOYMENT.md) and [MODEL_CARD.md](MODEL_CARD.md).

## Contract and spec (repo-side — complete)

- [x] `tabular_regression` semantics documented, independent of DIMER's `Custom / Other → object_detection` fallback; safe runtime override `DIMER_TASK_TYPE=tabular_regression` baked into the finetuner image — README / DEPLOYMENT / `TABULAR_REGRESSION_DATASET_SPEC.md`.
- [x] `dimer-pipeline.json` is the authoritative parameter contract; `model_id` is **not** duplicated as a hyperparameter (verified: absent from the manifest).
- [x] Base-model handoff specified — DEPLOYMENT "Weights delivery"; requested id, resolved revision, and loaded artifact recorded in `provenance`.
- [x] Result/provenance schema documented — MODEL_CARD; `weightsSha256`/`configSha256`, `baseModelRevision`, dataset SHA, AutoGluon version.
- [x] Regression result/provenance: negative targets/predictions accepted; `valEvaluation` error metrics reported as conventional positive values.
- [x] Smoke matrix includes negative targets and held-out `test.csv` scoring — `examples/build_freshretailnet_dataset.py`; regressor negatives covered.
- [x] Forecast/time-series example uses leakage-safe validation — purged per-series chronological split (`_temporal_splits`, embargo = horizon) applied before capping.
- [x] Cross-repo synchronization automated — `scripts/check_shared.py` enforces the shared dataset-resolution block byte-identical + against a cross-repo pinned SHA in every repo's CI.

## Real-stack verification (complete this session)

- [x] Build → offline Mitra load → fine-tune → save → reload → predict, exercised by `.github/workflows/integration.yml` (manual/nightly GPU) and run live on the 5070 Ti 2026-08-19.

## Release gate (open — platform-owned)

- [ ] Validator smoke run passes **inside DIMER**.
- [ ] Finetuner smoke run passes **inside DIMER**; base model recorded matches the model loaded.
- [ ] Saved `TabularPredictor` served by DIMER's inference-serving layer (tabular, not vision).
- [ ] All manifest controls confirmed to affect runtime on-platform.

**This is the only residual and it cannot be closed from these repos** — it needs the DIMER
portal. Tracked as a separate platform follow-up so closing #1 (repo contract) does not bury it.

`Closes #1` — repo-side contract complete; platform serving gate carried forward separately.
