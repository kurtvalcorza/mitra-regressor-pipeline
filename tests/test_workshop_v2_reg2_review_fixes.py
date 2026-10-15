"""Checks for the W14 review of the guided regression workshop (REG2-M1, REG2-M2, REG2-m1 … REG2-m11, 2026-10-12)
and the peer review of those fixes (evaluation status, exposure logging, 2026-10-14).

The dynamic tests execute the notebook's own cell code with stub session objects and small synthetic tables. They are
logic checks on CPU, not model runs or Colab execution evidence.
"""

from __future__ import annotations

import json
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "tutorials" / "DIMER_FreshRetailNet_MultiModel_Regression_Workshop_v2.ipynb"
CELLS = {cell["metadata"]["id"]: "".join(cell["source"]) for cell in json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]}
TARGET_EDA, FREEZE, TEST, TEST_TABLE, BOOTSTRAP, PREVIEW = (
    "4b2a6a68", "85d961d1", "ad332df3", "c3cc0309", "0944763fc02f", "uybWFVYNpD-1"
)
CLASSICAL_ABLATION, FM_ACTIVITY, FEATURE_EDA = "1f18af22", "e6bcf0b1", "91a6ce79"


class _Anything:
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


class _Session:
    """Stand-in for the notebook's Session: one experiment folder, a freeze flag and a development check."""

    def __init__(self, root: Path, experiment_id: str):
        self.root, self.experiment_id = root, experiment_id
        root.mkdir(parents=True, exist_ok=True)
        self.frozen = None

    def state(self):
        return {"freeze_sha256": self.frozen}

    def assert_development(self):
        if self.frozen:
            raise RuntimeError("This experiment is frozen. Start a new experiment before fitting or tuning again.")

    def freeze(self, keys, controls):
        (self.root / "freeze.json").write_text(json.dumps({"selected": keys}))
        self.frozen = "sha-" + self.experiment_id
        return self.root / "freeze.json"

    def validate_freeze(self, controls):
        return None


def _experiment(workspace: Path, experiment_id: str, prints: list) -> dict:
    validation = pd.DataFrame(
        {
            "model_key": ["lightgbm", "random_forest", "training_mean", "tabpfn3_icl", "tabiclv2_icl", "lightgbm_no_stockout"],
            "mae": [0.373, 0.376, 0.696, 0.371, 0.367, 0.360],
            "r2": [0.735, 0.740, 0.0, 0.692, 0.727, 0.70],
        }
    )
    namespace = {
        "pd": pd, "np": np, "json": json, "Path": Path, "print": lambda *a, **k: prints.append(" ".join(map(str, a))),
        "display": lambda *a, **k: None, "plt": _Anything(), "show_and_save_plot": lambda **k: None,
        "WORKSPACE_ROOT": workspace, "SESSION": _Session(workspace / experiment_id, experiment_id),
        "INCLUDE_TEST_TARGETS_IN_EDA": False, "TEST_TARGETS_VIEWED": False, "TARGET_COLUMN": "target",
        "frames": {split: pd.DataFrame({"target": np.arange(40.0)}) for split in ("train", "val", "test")},
        "CURRENT_CLASSICAL_KEYS": ["lightgbm", "random_forest", "training_mean"],
        "CURRENT_FOUNDATION_KEYS": ["tabpfn3_icl", "tabiclv2_icl"],
        "CURRENT_ABLATION_KEYS": ["lightgbm_no_stockout"],
        "CONDITION_SPECS": {"tabpfn3_icl": {}, "tabiclv2_icl": {}},
        "PRIMARY_METRIC": "mae", "baseline_validation_results": validation.iloc[:3], "combined_validation_results": validation,
        "parse_ablation_key": lambda key: ("lightgbm", "no_stockout") if key.endswith("_no_stockout") else None,
        "ablation_features": lambda group: [], "foundation_controls": lambda key, *a: {"backend": "foundation"},
        "classical_controls": lambda key, *a: {"backend": "classical"}, "assert_current_data": lambda: None,
        "write_json": lambda path, value: Path(path).write_text(json.dumps(value, allow_nan=False)),
        "normalize_results": lambda records: pd.DataFrame(records),
        "evaluate_classical_frozen": lambda *a: "receipt", "ENGINE": types.SimpleNamespace(run_test=lambda *a: "receipt"),
        "load_results": lambda *a, **k: (pd.DataFrame({"model_key": ["lightgbm"], "mae": [0.46]}), {}),
    }
    return _run(TARGET_EDA, namespace)


# --- REG2-M2: the validation-selected model is recorded at the freeze --------------------------------------------


def test_freeze_records_the_validation_selected_full_feature_model(tmp_path: Path) -> None:
    prints: list[str] = []
    ns = _run(FREEZE, _experiment(tmp_path, "exp-a", prints))
    record = json.loads((ns["SESSION"].root / "comparator.json").read_text())
    # The ablation scores 0.360 on validation but is not a full-feature condition, so TabICLv2 (0.367) is selected.
    assert record["validation_selected_key"] == "tabiclv2_icl"
    assert record["validation_selected_score"] == pytest.approx(0.367)
    assert "alphabetically first model_key" in record["validation_selected_rule"]
    assert record["reference_key"] == "lightgbm" and record["test_previously_exposed"] is False
    assert any("Validation-selected model" in line and "tabiclv2_icl" in line for line in prints)


def test_validation_ties_go_to_the_alphabetically_first_key(tmp_path: Path) -> None:
    ns = _experiment(tmp_path, "exp-a", [])
    ns["combined_validation_results"].loc[ns["combined_validation_results"].model_key.eq("tabpfn3_icl"), "mae"] = 0.367
    _run(FREEZE, ns)
    assert json.loads((ns["SESSION"].root / "comparator.json").read_text())["validation_selected_key"] == "tabiclv2_icl"


def _bootstrap(tmp_path: Path, primary: str | None, exposed: bool = False) -> tuple[dict, list[str]]:
    rng = np.random.default_rng(0)
    truth = rng.normal(size=80)
    predictions = {"lightgbm": truth + rng.normal(scale=0.5, size=80), "tabiclv2_icl": truth + rng.normal(scale=0.3, size=80),
                   "tabpfn3_icl": truth + rng.normal(scale=0.25, size=80)}
    record = {"reference_key": "lightgbm", "selection_basis": "fixture", "primary_metric": "mae"}
    if primary:
        record["validation_selected_key"] = primary
    if exposed:
        record["test_previously_exposed"] = True
    (tmp_path / "comparator.json").write_text(json.dumps(record))
    table = pd.DataFrame({"model_key": list(predictions), "model": [k.upper() for k in predictions], "condition": "fixture",
                          "mae": [float(np.mean(np.abs(truth - p))) for p in predictions.values()], "run_id": "r"})
    prints: list[str] = []
    ns = {"np": np, "pd": pd, "json": json, "plt": _Anything(), "display": lambda *a, **k: None,
          "print": lambda *a, **k: prints.append(" ".join(map(str, a))), "show_and_save_plot": lambda **k: None,
          "RANDOM_SEED": 0, "PRIMARY_METRIC": "mae", "TARGET_COLUMN": "target", "TEST_EVALUATION_COMPLETED": True,
          "frames": {"test": pd.DataFrame({"target": truth})}, "final_test_results": table,
          "foundation_test_predictions": {k: pd.DataFrame({"row_id": range(80), "target": truth, "prediction": p})
                                          for k, p in predictions.items()},
          "SESSION": types.SimpleNamespace(root=tmp_path),
          "parse_ablation_key": lambda key: ("lightgbm", "no_stockout") if key.endswith("_no_stockout") else None}
    return _run(BOOTSTRAP, ns), prints


def test_bootstrap_marks_the_validation_selected_row_as_confirmatory(tmp_path: Path) -> None:
    ns, prints = _bootstrap(tmp_path, "tabiclv2_icl")
    table = ns["BOOTSTRAP_RESULTS"].set_index("model_key")
    assert table["validation_selected"].to_dict() == {"lightgbm": False, "tabiclv2_icl": True, "tabpfn3_icl": False}
    # TabPFN-3 tops the test table in this fixture, but the confirmatory line names the validation-selected model.
    assert table["mae"].idxmin() == "tabpfn3_icl"
    assert any(line.startswith("Validation-selected result (confirmatory)") and "tabiclv2_icl" in line for line in prints)
    assert table["evaluation_status"].to_dict() == {
        "lightgbm": "descriptive", "tabiclv2_icl": "confirmatory", "tabpfn3_icl": "descriptive"}


def test_bootstrap_labels_the_selected_result_exploratory_after_prior_exposure(tmp_path: Path) -> None:
    ns, prints = _bootstrap(tmp_path, "tabiclv2_icl", exposed=True)
    table = ns["BOOTSTRAP_RESULTS"].set_index("model_key")
    assert table.loc["tabiclv2_icl", "evaluation_status"] == "exploratory (test previously exposed)"
    assert not any("(confirmatory)" in line for line in prints)
    assert any(line.startswith("Validation-selected result (exploratory (test previously exposed))") for line in prints)


def test_bootstrap_labels_intervals_illustrative_and_names_multiplicity(tmp_path: Path) -> None:
    _, prints = _bootstrap(tmp_path, None)
    text = "\n".join(prints)
    assert "illustrative row-level (IID) bootstrap" in text and "2 paired comparisons" in text
    assert "every row is descriptive" in text


def test_freeze_record_without_selection_still_scores(tmp_path: Path) -> None:
    ns, _ = _bootstrap(tmp_path, None)
    assert not ns["BOOTSTRAP_RESULTS"]["validation_selected"].any()


def test_reg2_m1_wording_is_descriptive() -> None:
    markdown = "\n".join(CELLS[c] for c in ("a28cc9746a30", "1cdbe62bb632", "391d4d6c"))
    assert "minimum amount of uncertainty" not in markdown
    assert "paired intervals above 0" not in markdown and "call a difference real" not in markdown
    assert "multiple-comparison" in CELLS["a28cc9746a30"] and "validation-selected" in CELLS["391d4d6c"]


# --- REG2-m7: test exposure survives a new experiment in the same runtime ------------------------------------------


def test_a_new_experiment_records_test_exposure_from_an_earlier_one(tmp_path: Path) -> None:
    first = _run(TEST, _run(FREEZE, _experiment(tmp_path, "exp-a", [])))
    assert first["TEST_EVALUATION_COMPLETED"] is True
    assert json.loads((first["SESSION"].root / "comparator.json").read_text())["prior_test_exposure"] == []
    # Start new experiment (0.3) resets every in-memory test flag; the log in the workspace remains.
    prints: list[str] = []
    second = _run(FREEZE, _experiment(tmp_path, "exp-b", prints))
    record = json.loads((second["SESSION"].root / "comparator.json").read_text())
    assert record["test_previously_exposed"] is True
    assert record["evaluation_status"] == "exploratory (test previously exposed)"
    events = [(e["event"], e["experiment_id"], e["freeze_sha256"]) for e in record["prior_test_exposure"]]
    assert events == [("test_scoring_attempted", "exp-a", "sha-exp-a"), ("test_scoring_completed", "exp-a", "sha-exp-a")]
    assert any("exploratory" in line for line in prints)


def test_unexposed_freeze_is_confirmatory(tmp_path: Path) -> None:
    ns = _run(FREEZE, _experiment(tmp_path, "exp-a", []))
    assert json.loads((ns["SESSION"].root / "comparator.json").read_text())["evaluation_status"] == "confirmatory"


def test_a_test_failure_after_scoring_started_still_counts_as_exposure(tmp_path: Path) -> None:
    ns = _run(FREEZE, _experiment(tmp_path, "exp-a", []))
    calls = []

    def evaluate(*args):
        calls.append(args[1])
        if len(calls) == 2:
            raise RuntimeError("model failed after an earlier test score was shown")
        return "receipt"

    ns["evaluate_classical_frozen"] = evaluate
    with pytest.raises(RuntimeError, match="model failed"):
        _run(TEST, ns)
    second = _run(FREEZE, _experiment(tmp_path, "exp-b", []))
    record = json.loads((second["SESSION"].root / "comparator.json").read_text())
    assert [e["event"] for e in record["prior_test_exposure"]] == ["test_scoring_attempted"]
    assert record["evaluation_status"] == "exploratory (test previously exposed)"


def test_test_target_relationship_in_3_6_is_logged_before_display(tmp_path: Path) -> None:
    ns = _experiment(tmp_path, "exp-a", [])
    shown = []
    ns.update({"feature_columns": ["stockout_hours"], "RANDOM_SEED": 0, "INCLUDE_TEST_TARGETS_IN_EDA": True,
               "display": lambda *a, **k: shown.append(len(ns["prior_test_exposure"]("exp-b")))})
    ns["frames"] = {split: pd.DataFrame({"stockout_hours": [0, 1, 2, 0] * 5, "target": np.arange(20.0)})
                    for split in ("train", "val", "test")}
    _run(FEATURE_EDA, ns, {'RELATIONSHIP_SPLIT = "train"': 'RELATIONSHIP_SPLIT = "test"'})
    assert shown and shown[0] == 1  # the entry existed before the first test-target table was displayed
    assert ns["TEST_TARGETS_VIEWED"] is True
    assert [e["cell"] for e in ns["prior_test_exposure"]("exp-b")] == ["3.6"]


def test_ablation_effects_compare_with_the_full_feature_parent(tmp_path: Path) -> None:
    ns, _ = _bootstrap(tmp_path, "tabiclv2_icl")
    assert "ABLATION_TEST_EFFECTS" not in ns  # no ablation in this fixture
    rng = np.random.default_rng(3)
    truth = rng.normal(size=50)
    preds = {"lightgbm": truth + 0.2, "lightgbm_no_stockout": truth + 0.3}
    record = {"reference_key": "lightgbm", "selection_basis": "f", "primary_metric": "mae"}
    (tmp_path / "comparator.json").write_text(json.dumps(record))
    table = pd.DataFrame({"model_key": list(preds), "model": ["LightGBM", "LightGBM without stockout"],
                          "condition": "f", "mae": [0.2, 0.3], "run_id": "r"})
    ns.update({"frames": {"test": pd.DataFrame({"target": truth})}, "final_test_results": table,
               "foundation_test_predictions": {k: pd.DataFrame({"row_id": range(50), "target": truth, "prediction": p})
                                               for k, p in preds.items()}})
    _run(BOOTSTRAP, ns)
    (effect,) = ns["ABLATION_TEST_EFFECTS"].to_dict("records")
    assert effect["full_feature_parent"] == "LightGBM" and effect["change_vs_parent"] == pytest.approx(0.1)


def test_test_target_eda_is_logged_too(tmp_path: Path) -> None:
    ns = _experiment(tmp_path, "exp-a", [])
    ns["INCLUDE_TEST_TARGETS_IN_EDA"] = True
    _run(TARGET_EDA, ns)
    assert [e["event"] for e in ns["prior_test_exposure"]("exp-b")] == ["test_targets_viewed_in_eda"]
    assert ns["prior_test_exposure"]("exp-a") == []


def test_export_manifest_carries_the_exposure_record() -> None:
    export = CELLS["e0f495f3"]
    assert '"prior_test_exposure":earlier_exposure' in export and '"validation_selected_key"' in export


# --- REG2-m2: Run all with Freeze now off reaches the end --------------------------------------------------------------


def test_freeze_off_run_all_skips_the_test_sections(tmp_path: Path) -> None:
    prints: list[str] = []
    ns = _experiment(tmp_path, "exp-a", prints)
    _run(FREEZE, ns, {"FREEZE_NOW = True": "FREEZE_NOW = False"})
    for cell in (TEST, TEST_TABLE, BOOTSTRAP, PREVIEW):
        _run(cell, ns)
    assert ns["TEST_EVALUATION_COMPLETED"] is False and ns["foundation_test_predictions"] == {}
    skipped = [line for line in prints if line.startswith("Skipped")]
    assert len(skipped) == 3 and not (ns["SESSION"].root / "freeze.json").exists()


# --- REG2-m5: rerunning 6.1 keeps the 6.2 foundation-model ablation ------------------------------------------------------


def _ablation_namespace(tmp_path: Path, frozen: bool) -> dict:
    session = _Session(tmp_path, "exp-a")
    session.frozen = "sha" if frozen else None
    return {"pd": pd, "feature_columns": [], "SESSION": session, "assert_current_data": lambda: None,
            "normalize_results": lambda records: pd.DataFrame(records), "CONDITION_SPECS": {"tabiclv2_icl": {}},
            "CURRENT_ABLATION_KEYS": ["lightgbm_no_stockout", "lightgbm_no_period_proxies", "tabiclv2_icl_no_stockout"],
            "ABLATION_RECEIPTS": [], "model_templates": {}, "print": lambda *a, **k: None}


def test_rerunning_classical_ablation_keeps_the_foundation_ablation(tmp_path: Path) -> None:
    ns = _run(CLASSICAL_ABLATION, _ablation_namespace(tmp_path, False), {"RUN_LOCAL_ABLATION = True": "RUN_LOCAL_ABLATION = False"})
    assert ns["CURRENT_ABLATION_KEYS"] == ["tabiclv2_icl_no_stockout"]


def test_rerunning_classical_ablation_in_a_frozen_experiment_leaves_the_selection(tmp_path: Path) -> None:
    ns = _ablation_namespace(tmp_path, True)
    before = list(ns["CURRENT_ABLATION_KEYS"])
    with pytest.raises(RuntimeError, match="This experiment is frozen"):
        _run(CLASSICAL_ABLATION, ns)
    assert ns["CURRENT_ABLATION_KEYS"] == before
    ns = _run(CLASSICAL_ABLATION, _ablation_namespace(tmp_path, True), {"RUN_LOCAL_ABLATION = True": "RUN_LOCAL_ABLATION = False"})
    assert ns["CURRENT_ABLATION_KEYS"] == before


# --- Prose and static findings ----------------------------------------------------------------------------------------------


def test_plots_follow_the_primary_metric() -> None:
    for cell in ("52a81ebd", TEST_TABLE):
        assert "[PRIMARY_METRIC])" in CELLS[cell] and ".mae)" not in CELLS[cell]


def test_activity_asks_for_a_configuration_run_all_did_not_print() -> None:
    assert "RUN_FOUNDATION_ABLATION = True" in CELLS["3ea12869"]
    assert "set `RUN_FOUNDATION_ABLATION=True`" not in CELLS[FM_ACTIVITY]
    assert 'FOUNDATION_ABLATION_GROUP="no_period_proxies"' in CELLS[FM_ACTIVITY]


def test_checkpoint_selection_metric_is_stated_and_matches_the_carried_code() -> None:
    assert "best checkpoint by validation MAE" in CELLS["dc3db879"]
    # EVAL14: the stated metric is the one the carried code uses for both fine-tuned models.
    assert 'eval_metric="mean_absolute_error"' in CELLS["45f3d66d"]  # Mitra (AutoGluon predictor), Section 0.3 core
    assert 'eval_metric="mae"' in CELLS["97391f5b"]  # TabICLv2 fine-tuner, Section 5.3 runner


def test_disclosures_references_and_troubleshooting() -> None:
    assert "fixed-recipe comparison" in CELLS["b31c69d7"] and "without hyperparameter search" in CELLS["391d4d6c"]
    assert "(Optional GPU extension.)" in CELLS["a7f7ada2"]
    assert "about 14 GB" in CELLS["1301aef25d05"] and "651.9 s" in CELLS["1301aef25d05"]
    for card in ("autogluon/mitra-regressor", "Layer6/TabDPT", "Prior-Labs/tabpfn_3", "jingang/TabICL"):
        assert f"huggingface.co/{card}/tree/" in CELLS["560086a7"]
    assert "This experiment is frozen. Start a new experiment" in CELLS["c40859f03f01"]
    assert "restores runtime state only" in CELLS["c40859f03f01"]


def test_peer_review_wording() -> None:
    assert "issued after day *t* closes" in CELLS["b1c60e9c"] and "known on or before" not in CELLS["b1c60e9c"]
    assert "pooled over rows" in CELLS["b1c60e9c"]
    assert "as it should" not in CELLS["770f915e"] and "R² ≤ 0" in CELLS["770f915e"]
    assert "cannot establish superiority" in CELLS["f3824bbd"]
    assert "illustrative IID 95% bootstrap intervals" in CELLS[BOOTSTRAP] and "illustrative IID" in CELLS["ee2fdb595a22"]
    assert "MAE by prediction-day stockout exposure" in CELLS["0de0df82"]
    assert "ALLOW_ADDITIONAL_FEATURES = False" in CELLS["2f62cb50"] and "unverified" in CELLS["54c9334a"]
    assert '"evaluation_status"' in CELLS["e0f495f3"] and "confirmatory / exploratory" in CELLS["391d4d6c"]


def test_notebook_declares_spec_2_5() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert notebook["metadata"]["dimer"]["notebook_spec"] == "2.5"
    assert "Notebook Specification 2.5" in CELLS["f79c68c5"] and "Specification 2.1" not in CELLS["aee8af32"]
    assert '"notebook_spec":"2.5"' in CELLS["e0f495f3"] and '"notebook_spec":"2.5"' in CELLS["wv3iBhT9pD-2"]
