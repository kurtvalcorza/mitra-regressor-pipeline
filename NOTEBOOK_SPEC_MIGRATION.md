# Notebook Specification 1.0 migration

This branch migrates the Mitra Regressor tutorial pair to DIMER Notebook Specification 1.0.

## Profiles

- `tutorials/mitra_regressor_colab.ipynb` — `E2E`
- `tutorials/mitra_regressor_predictor_inference_colab.ipynb` — `ARTIFACT-INFERENCE`

## Implemented requirements

- repository-owned public tutorial API for validation, fitting, prediction, model/package verification, archive safety, and artifact validation;
- immutable Mitra model identity, revision, and SHA-256 verification;
- DIMER offline package manifest validation;
- executable mean/median constant baselines and regression point-estimate/uncertainty guidance;
- split-overlap reporting and production-facing CSV validation;
- versioned predictor artifact metadata plus file manifest with size/SHA-256 checks;
- safe archive extraction with traversal, backslash, symlink, per-member, expanded-size, and compression-ratio guards;
- fresh-directory reload and numerical equivalence verification;
- externally supplied artifact consumption by the `ARTIFACT-INFERENCE` notebook;
- exact direct tutorial dependency pins;
- static Notebook Specification conformance checks and API/security regression tests;
- separate clean-runtime notebook release execution workflow.

## Release evidence

Static checks are not treated as execution evidence. The pull-request workflow `Notebook release execution` must execute the E2E default path on a clean hosted runner, produce `mitra-predictor.zip`, and then execute the artifact-inference notebook against that externally produced archive with its SHA-256 supplied. The resulting executed notebooks and predictor archive are retained as workflow artifacts.
