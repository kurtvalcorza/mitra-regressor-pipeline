"""Acceptance checks for the Notebook Review Framework v1 regression findings (REG-01 … REG-05).

Each test executes the notebook's own cell code against small synthetic inputs, with only the heavyweight notebook
state stubbed. These are logic checks, not model runs or Colab execution evidence.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "tutorials" / "DIMER_FreshRetailNet_MultiModel_Regression_Workshop_v2.ipynb"
CELLS = {
    cell.get("id") or cell["metadata"]["id"]: "".join(cell["source"])
    for cell in json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
}


class _Anything:
    """Absorbs matplotlib calls in cells whose figures are not under test."""

    def __getattr__(self, name):
        return self

    def __call__(self, *args, **kwargs):
        return self


def _run(cell_id: str, namespace: dict, replacements: dict[str, str] | None = None) -> dict:
    source = CELLS[cell_id]
    for old, new in (replacements or {}).items():
        assert source.count(old) == 1, old
        source = source.replace(old, new)
    exec(compile(source, cell_id, "exec"), namespace)
    return namespace


def _write_json(path, value) -> None:
    Path(path).write_text(json.dumps(value, allow_nan=False))


# --- REG-01: the comparator is fixed at the freeze; a foundation-only selection is supported --------------------


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


# --- REG-01 / REG-02 / REG-05: the bootstrap cell --------------------------------------------------------------


def _bootstrap(tmp_path: Path, metric: str, truth, predictions: dict, reference) -> dict:
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
        "plt": _Anything(),
        "display": lambda *a, **k: None,
        "show_and_save_plot": lambda **k: None,
        "RANDOM_SEED": 0,
        "PRIMARY_METRIC": metric,
        "TARGET_COLUMN": "target",
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


# --- REG-03: the three-band exercise refuses a collapsed derived task -------------------------------------------


@pytest.fixture
def balanced_accuracy(monkeypatch):
    try:
        import sklearn.metrics  # noqa: F401
    except ModuleNotFoundError:  # the CI image has no scikit-learn; provide the one function the cell imports

        def score(truth, pred):
            truth, pred = np.asarray(truth), np.asarray(pred)
            return float(np.mean([np.mean(pred[truth == c] == c) for c in np.unique(truth)]))

        metrics = types.ModuleType("sklearn.metrics")
        metrics.balanced_accuracy_score = score
        monkeypatch.setitem(sys.modules, "sklearn", types.ModuleType("sklearn"))
        monkeypatch.setitem(sys.modules, "sklearn.metrics", metrics)


def _bands(train, val, test, prediction) -> dict:
    frames = {name: pd.DataFrame({"target": np.asarray(v, dtype=float)}) for name, v in (("train", train), ("val", val), ("test", test))}
    namespace = {
        "np": np,
        "pd": pd,
        "display": lambda *a, **k: None,
        "frames": frames,
        "TARGET_COLUMN": "target",
        "USE_BYOD": True,
        "TEST_EVALUATION_COMPLETED": True,
        "validation_predictions": {"constant": np.full(len(val), prediction)},
        "prediction_registry": {},
        "foundation_test_predictions": {"constant": pd.DataFrame({"prediction": np.full(len(test), prediction)})},
        "combined_validation_results": pd.DataFrame({"model_key": ["constant"], "model": ["Constant"], "condition": ["fixture"]}),
    }
    return _run("5e0e1cbe725c", namespace)


def test_tied_targets_do_not_score_as_three_bands(balanced_accuracy) -> None:
    out = _bands([0.0] * 70 + [1.0] * 20 + [2.0] * 10, [0.0, 1.0, 2.0], [0.0, 1.0, 2.0], 1.0)
    assert out["BANDS_VALID"] is False and out["REGRESSION_AS_BANDS"].empty


def test_ordinary_targets_score_a_one_band_guess_at_one_third(balanced_accuracy) -> None:
    out = _bands([0.0, 1.0, 2.0] * 40, [0.0, 1.0, 2.0] * 3, [0.0, 1.0, 2.0] * 3, 0.0)
    assert out["BANDS_VALID"] is True
    assert out["REGRESSION_AS_BANDS"]["test_balanced_accuracy"].iloc[0] == pytest.approx(1 / 3)


# --- REG-04 ----------------------------------------------------------------------------------------------------


def test_worked_answers_describe_a_reference_configuration() -> None:
    answers = CELLS["1cdbe62bb632"]
    assert "do not change between revisions" not in answers
    assert "one reference run" in answers
    conclusion = CELLS["391d4d6c"]
    assert "primary metric chosen in Section 0.2" in conclusion and "obtained an MAE of" not in conclusion
