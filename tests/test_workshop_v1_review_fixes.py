"""Acceptance checks for the 2026-10-03 Notebook Review Framework v1 findings on the v1 regression workshop (FRR1-*).

Each dynamic test executes the notebook's own cell code against small synthetic inputs, with only the heavyweight
notebook state stubbed. These are logic checks, not model runs or Colab execution evidence.
"""

from __future__ import annotations

import ast
import json
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "tutorials" / "DIMER_FreshRetailNet_MultiModel_Regression_Workshop.ipynb"
NB = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
CELLS = {cell["id"]: "".join(cell["source"]) for cell in NB["cells"]}
MARKDOWN = "\n".join("".join(cell["source"]) for cell in NB["cells"] if cell["cell_type"] == "markdown")
README = (ROOT / "tutorials" / "README.md").read_text(encoding="utf-8")


class _Recorder:
    """Absorbs matplotlib calls and records plot titles."""

    def __init__(self):
        self.titles: list[str] = []

    def title(self, text, *args, **kwargs):
        self.titles.append(text)

    def __getattr__(self, name):
        return lambda *args, **kwargs: self


def _run(cell_id: str, namespace: dict, replacements: dict[str, str] | None = None) -> dict:
    source = CELLS[cell_id]
    for old, new in (replacements or {}).items():
        assert source.count(old) == 1, old
        source = source.replace(old, new)
    exec(compile(source, cell_id, "exec"), namespace)
    return namespace


def _write_json(path, value) -> None:
    Path(path).write_text(json.dumps(value, allow_nan=False))


# --- FRR1-M2 / FRR1-M3: the comparator is fixed at the freeze -------------------------------------------------


def _freeze(tmp_path: Path, selected: str) -> dict:
    baselines = pd.DataFrame(
        {"model_key": ["lightgbm", "random_forest", "training_mean"], "mae": [0.373, 0.380, 0.6], "r2": [0.7, 0.72, 0.0]}
    )
    session = types.SimpleNamespace(root=tmp_path, freeze=lambda keys, controls: {"selected": keys})
    namespace = {
        "CURRENT_CLASSICAL_KEYS": ["lightgbm", "random_forest", "training_mean"],
        "CURRENT_FOUNDATION_KEYS": ["tabiclv2_icl", "mitra_icl"],
        "CURRENT_ABLATION_KEYS": [],
        "CONDITION_SPECS": {"tabiclv2_icl": {}, "mitra_icl": {}},
        "INCLUDE_TEST_TARGETS_IN_EDA": False,
        "TEST_TARGETS_VIEWED": False,
        "SESSION": session,
        "PRIMARY_METRIC": "mae",
        "baseline_validation_results": baselines,
        "frames": {"test": pd.DataFrame({"target": np.arange(40.0)})},
        "parse_ablation_key": lambda key: None,
        "foundation_controls": lambda key, *a: {"backend": "foundation"},
        "classical_controls": lambda key, *a: {"backend": "classical"},
        "assert_current_data": lambda: None,
        "write_json": _write_json,
    }
    return _run("85d961d1", namespace, {'SELECTED_MODEL_KEYS = "" # @param': f'SELECTED_MODEL_KEYS = "{selected}" # @param'})


def test_freeze_records_the_validation_best_frozen_baseline(tmp_path: Path) -> None:
    _freeze(tmp_path, "tabiclv2_icl,random_forest,lightgbm")
    recorded = json.loads((tmp_path / "comparator.json").read_text())
    assert recorded["reference_key"] == "lightgbm" and recorded["primary_metric"] == "mae"


def test_foundation_only_freeze_records_no_comparator(tmp_path: Path) -> None:
    _freeze(tmp_path, "tabiclv2_icl,mitra_icl")
    assert json.loads((tmp_path / "comparator.json").read_text())["reference_key"] is None


# --- FRR1-M2 / FRR1-M3 / FRR1-m1 / FRR1-m3: the bootstrap cell -------------------------------------------------


def _bootstrap(tmp_path: Path, metric: str, truth, predictions: dict, reference, completed: bool = True) -> dict:
    (tmp_path / "comparator.json").write_text(
        json.dumps({"reference_key": reference, "selection_basis": "test fixture", "primary_metric": metric})
    )
    truth = np.asarray(truth, dtype=float)

    def score(pred):
        error = truth - pred
        if metric == "mae":
            return np.mean(np.abs(error))
        if metric == "rmse":
            return np.sqrt(np.mean(error**2))
        with np.errstate(invalid="ignore", divide="ignore"):
            return 1 - np.sum(error**2) / np.sum((truth - truth.mean()) ** 2)

    frames = {
        key: pd.DataFrame({"row_id": range(len(truth)), "target": truth, "prediction": np.asarray(pred, dtype=float)})
        for key, pred in predictions.items()
    }
    table = pd.DataFrame(
        {
            "model_key": list(predictions),
            "model": [key.upper() for key in predictions],
            "condition": ["fixture"] * len(predictions),
            metric: [score(np.asarray(p, dtype=float)) for p in predictions.values()],
            "run_id": ["r"] * len(predictions),
        }
    )
    namespace = {
        "np": np,
        "pd": pd,
        "json": json,
        "plt": _Recorder(),
        "display": lambda *a, **k: None,
        "show_and_save_plot": lambda **k: None,
        "RANDOM_SEED": 0,
        "PRIMARY_METRIC": metric,
        "TARGET_COLUMN": "target",
        "TEST_EVALUATION_COMPLETED": completed,
        "frames": {"test": pd.DataFrame({"target": truth})},
        "final_test_results": table,
        "foundation_test_predictions": frames,
        "SESSION": types.SimpleNamespace(root=tmp_path),
    }
    return _run("0944763fc02f", namespace)


def test_foundation_only_selection_keeps_individual_intervals(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    truth = rng.normal(size=60)
    out = _bootstrap(tmp_path, "mae", truth, {"tabiclv2_icl": truth + 0.1, "mitra_icl": truth - 0.2}, None)
    table = out["BOOTSTRAP_RESULTS"]
    assert table["ci_low"].notna().all() and table["gain_vs_reference"].isna().all()
    assert set(table["comparator"]) == {"none selected"}


@pytest.mark.parametrize("metric", ["mae", "rmse", "r2"])
def test_reported_gain_is_the_observed_score_difference(tmp_path: Path, metric: str) -> None:
    rng = np.random.default_rng(2)
    truth = rng.normal(size=80)
    predictions = {"lightgbm": truth + rng.normal(scale=0.5, size=80), "tabiclv2_icl": truth + rng.normal(scale=0.3, size=80)}
    table = _bootstrap(tmp_path, metric, truth, predictions, "lightgbm")["BOOTSTRAP_RESULTS"].set_index("model_key")
    model, reference = table.loc["tabiclv2_icl", metric], table.loc["lightgbm", metric]
    expected = model - reference if metric == "r2" else reference - model
    assert table.loc["tabiclv2_icl", "gain_vs_reference"] == pytest.approx(expected, abs=1e-12)
    assert table.loc["tabiclv2_icl", "comparator"] == "LIGHTGBM"


def test_r2_intervals_are_unavailable_when_draws_are_degenerate(tmp_path: Path) -> None:
    out = _bootstrap(tmp_path, "r2", [0.0, 1.0], {"lightgbm": [0.1, 0.9], "tabiclv2_icl": [0.2, 0.7]}, "lightgbm")
    table = out["BOOTSTRAP_RESULTS"].set_index("model_key")
    assert table.loc["lightgbm", "r2"] == pytest.approx(0.96)
    assert table["interval_status"].str.startswith("unavailable").all()
    assert table[["ci_low", "ci_high", "gain_ci_low", "gain_ci_high"]].isna().all().all()


def test_r2_on_a_constant_test_target_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="R² is undefined"):
        _bootstrap(tmp_path, "r2", [1.0, 1.0, 1.0], {"lightgbm": [1.0, 1.0, 1.0]}, "lightgbm")


def test_bootstrap_is_skipped_without_a_test_evaluation(tmp_path: Path) -> None:
    out = _bootstrap(tmp_path, "mae", [0.0, 1.0, 2.0], {"lightgbm": [0.0, 1.0, 2.0]}, "lightgbm", completed=False)
    assert "BOOTSTRAP_RESULTS" not in out


# --- FRR1-m3: an exploratory Run all (Freeze now off) does not stop at Section 8 --------------------------------


def test_test_evaluation_is_skipped_when_not_frozen(tmp_path: Path, capsys) -> None:
    namespace = {
        "SESSION": types.SimpleNamespace(root=tmp_path),
        "normalize_results": lambda rows: pd.DataFrame(rows),
        "foundation_test_predictions": {"stale": object()},
    }
    out = _run("ad332df3", namespace)
    assert out["TEST_EVALUATION_COMPLETED"] is False and out["foundation_test_predictions"] == {}
    assert "Skipped" in capsys.readouterr().out


def test_inference_preview_is_skipped_without_a_test_evaluation() -> None:
    _run("uybWFVYNpD-1", {"TEST_EVALUATION_COMPLETED": False})


def test_export_does_not_require_a_test_evaluation() -> None:
    export = CELLS["e0f495f3"]
    assert "Complete frozen-test evaluation first" not in export
    assert "EXPORT_TEST_RESULTS=INCLUDE_FROZEN_TEST_RESULTS and TEST_EVALUATION_COMPLETED" in export
    assert '"includes_test_results":EXPORT_TEST_RESULTS' in export
    assert 'shutil.copy2(SESSION.root / "comparator.json",export_directory / "comparator.json")' in export
    assert '"notebook_revision":"3.2.0"' in export


# --- FRR1-m2: plots and the conclusion template follow the primary metric -----------------------------------------


@pytest.mark.parametrize("metric", ["mae", "rmse", "r2"])
def test_result_plots_follow_the_primary_metric(metric: str) -> None:
    table = pd.DataFrame(
        {"model_key": ["a"], "model": ["A"], "condition": ["c"], "mae": [0.4], "rmse": [0.5],
         "median_absolute_error": [0.3], "r2": [0.6], "runtime_seconds": [1.0], "device": ["cpu"], "run_id": ["r"]}
    )
    common = {"pd": pd, "PRIMARY_METRIC": metric, "normalize_results": lambda frame: frame, "display": lambda *a, **k: None,
              "show_and_save_plot": lambda **k: None}
    validation_plot = _Recorder()
    _run("52a81ebd", {**common, "plt": validation_plot, "baseline_validation_results": table,
                      "foundation_validation_results": table.iloc[0:0]})
    test_plot = _Recorder()
    _run("c3cc0309", {**common, "plt": test_plot, "final_test_results": table})
    assert validation_plot.titles == [f"Validation {metric.upper()}: classical and foundation-model conditions"]
    assert test_plot.titles == [f"Frozen-model test {metric.upper()} — no refitting"]


def test_conclusion_template_uses_the_primary_metric_and_one_partition() -> None:
    conclusion = CELLS["391d4d6c"]
    assert "obtained an MAE of" not in conclusion and "primary metric chosen in Section 0.2" in conclusion
    assert conclusion.count("validation **[primary metric]**") == 2 and "test MAE" not in conclusion


# --- FRR1-m4: stockout bands keep zero-stockout rows separate; numbers in prose match the data ------------------


def test_stockout_bands_keep_zero_stockout_rows_separate() -> None:
    hours = np.array([0.0] * 6 + [1.0, 3.0, 5.0, 8.0, 16.0, 0.0])
    registry = {"m": pd.DataFrame({"target": np.linspace(0, 1, 12), "prediction": np.linspace(0, 1, 12) + 0.1,
                                   "stockout_hours": hours})}
    shown = []
    namespace = {
        "np": np, "pd": pd, "plt": _Recorder(), "PRIMARY_METRIC": "mae",
        "display": lambda frame, *a, **k: shown.append(frame),
        "show_and_save_plot": lambda **k: None,
        "choose_diagnostic_model": lambda registry, key, ranking: "m",
        "prediction_registry": registry,
        "foundation_validation_results": pd.DataFrame({"model_key": ["m"], "mae": [0.1]}),
        "SESSION": types.SimpleNamespace(current_record=lambda key: (None, {"run_id": "r"})),
    }
    out = _run("0de0df82", namespace)
    bands = out["stockout_error"]["stockout_band"].astype(str).tolist()
    assert bands == ["0 hours", "more than 0, up to 5 hours", "more than 5 hours"]


def test_expected_output_text_matches_what_the_cells_show() -> None:
    assert "the local archive path" not in CELLS["7ab366aa"]
    assert "Approximately 3.4% of target observations are zero-valued" not in MARKDOWN
    assert "3.4% of the training rows" in CELLS["758aa499"]


# --- FRR1-m7: the metric code shown in 4.1 is checked against the implementation behind the tables --------------


def _core_module() -> types.ModuleType:
    tree = ast.parse(CELLS["45f3d66d"])
    source = next(
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "CORE_SOURCE" for t in node.targets)
    )
    module = types.ModuleType("workshop_core")
    exec(compile(source, "workshop_core", "exec"), module.__dict__)
    return module


def test_illustrated_metrics_match_the_pipeline_metrics(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "workshop_core", _core_module())
    rng = np.random.default_rng(3)
    y_train, y_val = pd.Series(rng.gamma(2.0, 0.5, 50)), pd.Series(rng.gamma(2.0, 0.5, 30))

    def r2(y, p):
        return 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)

    namespace = {
        "np": np, "pd": pd, "Any": object, "display": lambda *a, **k: None, "y_train": y_train, "y_val": y_val,
        "mean_absolute_error": lambda y, p: np.mean(np.abs(y - p)),
        "mean_squared_error": lambda y, p: np.mean((y - p) ** 2),
        "median_absolute_error": lambda y, p: np.median(np.abs(y - p)),
        "r2_score": r2,
    }
    out = _run("9bc4fae6", namespace)
    assert "make_result_record" not in out and "Illustration" in CELLS["9bc4fae6"].splitlines()[0]


# --- FRR1-M1 / FRR1-M4 / FRR1-M5 / FRR1-m5 / FRR1-m6: learner-facing text ------------------------------------------


def test_recovery_after_a_frozen_run_is_documented() -> None:
    assert "Leave **Freeze now** off while exploring" not in MARKDOWN
    for cell_id in ("9567d1e8", "ad3ff624", "frr1howtouse", "frr1troubleshoot"):
        text = CELLS[cell_id]
        assert "Start new experiment" in text and "Section 1.1" in text and "Run after" in text, cell_id


def test_fine_tuning_question_is_marked_optional_with_its_switches() -> None:
    assert "*(Optional GPU extension.)* Does task-specific fine-tuning improve performance" in CELLS["a7f7ada2"]
    assert "*(Optional GPU extension.)* Did a fine-tuned condition actually update weights?" in CELLS["f3824bbd"]
    extension = CELLS["dc3db879"]
    for needle in ("RUN_MITRA_FINETUNED", "RUN_TABICLV2_FINETUNED", "600 seconds", "`effective_mode`", "not tested"):
        assert needle in extension, needle


def test_identity_edition_and_guided_layer_waivers() -> None:
    opening = CELLS["f79c68c5"]
    assert opening.startswith("# DIMER Notebook: ") and "**Compact workshop edition (v1" in opening
    assert "_v2.ipynb" in opening and "sample answers" in opening
    assert "(an estimate, not measured for this revision)" in opening
    assert "deliberate" in CELLS["7fcacb12"] and "same" in CELLS["7fcacb12"]
    assert "### Input → Model → Output" in CELLS["frr1howtouse"] and "**Infrastructure:**" in CELLS["frr1howtouse"]
    assert CELLS["frr1troubleshoot"].startswith("# Troubleshooting")
    assert NB["metadata"]["workshop_revision"] == "3.2.0"


def test_readme_says_which_notebook_to_use() -> None:
    assert "### Which notebook should I use?" in README
    assert "the same pipeline and defaults" not in README
    assert "Revision 3.2.0" in README
