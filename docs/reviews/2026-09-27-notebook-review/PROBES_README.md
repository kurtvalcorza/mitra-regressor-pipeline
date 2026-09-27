# Offline regression-notebook review probes

Target: `kurtvalcorza/mitra-regressor-pipeline`, commit `dd7c8ecd49ad1c7de4cae5875152a71c40f431d7`, notebook blob `026741c58e5a9557d14320f72239afc411f2351a`.

These probes manually transcribe the small reference-selection, R² scoring, and tertile-band expressions inspected in sections 8.0, 8.3, and 9.1. They replace model results and data with synthetic fixtures. They do **not** run the notebook, hosted Colab, its Session/LocalExperiment machinery, model adapters, or model weights. They make no claim about successful full default or BYOD execution.

Run `python review_probes.py` in an environment with NumPy, pandas, and scikit-learn. It overwrites the adjacent `probe_results.json` with actual observations and versions. Three reproduced bad-path outcomes and three positive controls are expected. A single-class warning from scikit-learn in the degenerate-band probe is expected; that warning is not an explanation of the notebook's changed three-band semantics.

The output records the environment used during this review. No new package installation or model download is required. Dependencies are not bundled.
