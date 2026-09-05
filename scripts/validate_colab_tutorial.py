#!/usr/bin/env python3
"""Static checks for the standalone Mitra Regressor Colab tutorials."""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "tutorials" / "mitra_regressor_colab.ipynb"
INFERENCE_NOTEBOOK = ROOT / "tutorials" / "mitra_regressor_predictor_inference_colab.ipynb"
TUTORIAL_README = ROOT / "tutorials" / "README.md"

PINNED_REVISION = "5f277aa8f69042d39d6ac3612aed18bb9279bd95"
WEIGHTS_SHA256 = "d8e75c62af0bec2fd404b0ad20a442d951d43ca6d331315cfcc0509b54f2c642"
CONFIG_SHA256 = "2bc1ed5047f7c25368245e8ad32540a5fa28940b1ec05d3f1f454a09ff5384c1"
SAMPLE_REVISION = "5625a9eeca94b8c72b9ad1ec78d07ecbaa720903"

REPO_INTERNAL_IMPORT_PARTS = {"finetuner", "validator"}
FORBIDDEN_TRAINING = (
    "autogluon/mitra-classifier",
    "freshretailnet-band-h7.zip",
    "predict_proba(",
    "probability_",
    "NUM_CLASSES",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def code_sources(cells: list[dict]) -> list[tuple[int, str]]:
    result = []
    for i, cell in enumerate(cells):
        if cell.get("cell_type") != "code":
            continue
        source = cell.get("source", "")
        if isinstance(source, list):
            source = "".join(source)
        source = "\n".join(
            line for line in str(source).splitlines()
            if not line.lstrip().startswith(("%", "!"))
        )
        result.append((i, source))
    return result


def load_notebook(path: Path) -> tuple[list[dict], str, list[tuple[int, str]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    require(payload.get("nbformat") == 4, f"{path.name} must use nbformat 4")
    cells = payload.get("cells", [])
    require(bool(cells), f"{path.name} has no cells")
    text = "\n".join(
        "".join(cell.get("source", []))
        if isinstance(cell.get("source", []), list)
        else str(cell.get("source", ""))
        for cell in cells
    )
    parsed = code_sources(cells)
    for i, source in parsed:
        if source.strip():
            ast.parse(source, filename=f"{path.name}:cell-{i}")
    return cells, text, parsed


def top_level_literal_assignments_match(code_cells, name, expected):
    found = False
    for _, source in code_cells:
        if not source.strip():
            continue
        tree = ast.parse(source)
        for node in tree.body:
            targets = []
            value = None
            if isinstance(node, ast.Assign):
                targets = node.targets
                value = node.value
            elif isinstance(node, ast.AnnAssign):
                targets = [node.target]
                value = node.value
            else:
                continue
            if not any(isinstance(t, ast.Name) and t.id == name for t in targets):
                continue
            found = True
            if not isinstance(value, ast.Constant):
                return False
            if type(value.value) is not type(expected) or value.value != expected:
                return False
    return found


def repo_internal_imports(code_cells):
    found = set()
    for _, source in code_cells:
        if not source.strip():
            continue
        for node in ast.walk(ast.parse(source)):
            modules = []
            if isinstance(node, ast.Import):
                modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules.append(node.module)
            for module in modules:
                if REPO_INTERNAL_IMPORT_PARTS.intersection(module.split(".")):
                    found.add(module)
    return found


def call_attributes(code_cells):
    attrs = set()
    for _, source in code_cells:
        if not source.strip():
            continue
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                attrs.add(node.func.attr)
    return attrs


def has_memory_guard(code_cells):
    for _, source in code_cells:
        if not source.strip():
            continue
        for node in ast.walk(ast.parse(source)):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "fit"
            ):
                continue
            for keyword in node.keywords:
                if keyword.arg != "ag_args_fit" or not isinstance(keyword.value, ast.Dict):
                    continue
                for key, value in zip(keyword.value.keys, keyword.value.values):
                    if (
                        isinstance(key, ast.Constant)
                        and key.value == "max_memory_usage_ratio"
                        and isinstance(value, ast.Name)
                        and value.id == "MAX_MEMORY_USAGE_RATIO"
                    ):
                        return True
    return False


def has_current_fit_completion_gate(code_cells):
    for _, source in code_cells:
        if not source.strip():
            continue
        tree = ast.parse(source)
        false_positions, fit_positions, true_positions = [], [], []
        for position, node in enumerate(tree.body):
            if isinstance(node, ast.Assign):
                names = [t.id for t in node.targets if isinstance(t, ast.Name)]
                if "FIT_RUN_COMPLETED" in names and isinstance(node.value, ast.Constant):
                    if node.value.value is False:
                        false_positions.append(position)
                    elif node.value.value is True:
                        true_positions.append(position)
                if (
                    isinstance(node.value, ast.Call)
                    and isinstance(node.value.func, ast.Name)
                    and node.value.func.id == "fit_mitra"
                ):
                    fit_positions.append(position)
        if false_positions and fit_positions and true_positions and min(false_positions) < min(fit_positions) < max(true_positions):
            return True
    return False


def fit_completion_gate_uses(code_cells):
    count = 0
    for _, source in code_cells:
        if not source.strip():
            continue
        for node in ast.walk(ast.parse(source)):
            if not isinstance(node, ast.Call):
                continue
            if not (isinstance(node.func, ast.Attribute) and node.func.attr == "get"):
                continue
            owner = node.func.value
            if not (isinstance(owner, ast.Call) and isinstance(owner.func, ast.Name) and owner.func.id == "globals"):
                continue
            if not node.args:
                continue
            if isinstance(node.args[0], ast.Constant) and node.args[0].value == "FIT_RUN_COMPLETED":
                count += 1
    return count


def inference_step_is_self_contained(code_cells):
    for _, source in code_cells:
        if "RUN_NEW_DATA_INFERENCE" not in source:
            continue
        tree = ast.parse(source)
        imported_io = imported_pd = imported_csv = False
        for node in tree.body:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported_io |= alias.name == "io"
                    imported_csv |= alias.name == "csv"
                    imported_pd |= alias.name == "pandas" and alias.asname == "pd"
        return imported_io and imported_csv and imported_pd
    return False


def has_safe_direct_weights_copy_guard(code_cells):
    def resolved_name(node):
        if not (
            isinstance(node, ast.Call)
            and not node.args
            and not node.keywords
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "resolve"
            and isinstance(node.func.value, ast.Name)
        ):
            return None
        return node.func.value.id

    for _, source in code_cells:
        if "weights_from_dimer" not in source:
            continue
        tree = ast.parse(source)
        for function in ast.walk(tree):
            if not isinstance(function, ast.FunctionDef) or function.name != "weights_from_dimer":
                continue
            for candidate in ast.walk(function):
                if not isinstance(candidate, ast.If):
                    continue
                test = candidate.test
                if not (
                    isinstance(test, ast.Compare)
                    and len(test.ops) == 1
                    and isinstance(test.ops[0], ast.NotEq)
                    and len(test.comparators) == 1
                ):
                    continue
                left_name = resolved_name(test.left)
                right_name = resolved_name(test.comparators[0])
                if not left_name or not right_name or left_name == right_name:
                    continue
                for node in ast.walk(candidate):
                    if not (
                        isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Attribute)
                        and isinstance(node.func.value, ast.Name)
                        and node.func.value.id == "shutil"
                        and node.func.attr == "copy2"
                        and len(node.args) >= 2
                        and isinstance(node.args[0], ast.Name)
                        and isinstance(node.args[1], ast.Name)
                    ):
                        continue
                    if node.args[0].id == left_name and node.args[1].id == right_name:
                        return True
    return False


def mitra_metric_map_is_regression(code_cells):
    expected = {"mean_absolute_error": "mae", "root_mean_squared_error": "rmse"}
    for _, source in code_cells:
        if not source.strip():
            continue
        tree = ast.parse(source)
        for node in tree.body:
            if not isinstance(node, ast.Assign):
                continue
            if not any(isinstance(t, ast.Name) and t.id == "MITRA_METRIC_MAP" for t in node.targets):
                continue
            if not isinstance(node.value, ast.Dict):
                return False
            actual = {}
            for key, value in zip(node.value.keys, node.value.values):
                if not (isinstance(key, ast.Constant) and isinstance(value, ast.Constant)):
                    return False
                actual[key.value] = value.value
            return all(actual.get(k) == v for k, v in expected.items())
    return False


def tabular_predictor_is_regression(code_cells):
    found = False
    for _, source in code_cells:
        if not source.strip():
            continue
        for node in ast.walk(ast.parse(source)):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "TabularPredictor"):
                continue
            found = True
            keyword = next((k for k in node.keywords if k.arg == "problem_type"), None)
            if not (keyword and isinstance(keyword.value, ast.Constant) and keyword.value.value == "regression"):
                return False
    return found


def metadata_records_regression(code_cells):
    for _, source in code_cells:
        if not source.strip():
            continue
        for node in ast.walk(ast.parse(source)):
            if not isinstance(node, ast.Dict):
                continue
            for key, value in zip(node.keys, node.values):
                if (
                    isinstance(key, ast.Constant) and key.value == "problem_type"
                    and isinstance(value, ast.Constant) and value.value == "regression"
                ):
                    return True
    return False


def direct_tabular_predictor_construction(code_cells):
    for _, source in code_cells:
        if not source.strip():
            continue
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "TabularPredictor":
                return True
    return False


def validate_training_tutorial():
    _, text, parsed_code = load_notebook(NOTEBOOK)
    required = (
        PINNED_REVISION, WEIGHTS_SHA256, CONFIG_SHA256, SAMPLE_REVISION,
        "autogluon.tabular[mitra]==1.5.0",
        "autogluon/mitra-regressor",
        "DIMER ZIP", "Pinned upstream",
        "freshretailnet-h7.zip", "DATASET_CARD.md",
        "Upload pre-split train/val/test",
        "target must be numeric",
        "target contains infinite values",
        "training target has no variation",
        "Capped training split has no target variation",
        "REG_LOWER_IS_BETTER",
        "MITRA_METRIC_MAP",
        "Unsupported EVAL_METRIC",
        "contains duplicate column names",
        "Recommended predictor for inference/export",
        "Holdout is too small for automatic model selection",
        "default:pretrained",
        "already contains a 'prediction' column",
        "Inference CSV contains duplicate column names",
        "FIT_RUN_COMPLETED",
        "No predictor was successfully trained in this Step 4 execution.",
        "Cleared stale predictor state and output paths before fitting.",
        "Reload smoke test failed",
        "reproduces smoke-test predictions",
        "TabularPredictor.load(str(RELOAD_DIR))",
        "np.allclose",
        "fine_tune_steps_requested",
        "GPT-5.6 Sol High",
        "OpenAI / ChatGPT",
        "Agent Relay role",
        "provenance, not sign-off",
    )
    for marker in required:
        require(marker in text, f"missing required tutorial marker: {marker}")

    for forbidden in FORBIDDEN_TRAINING:
        require(forbidden not in text, f"classification-only logic leaked into regressor tutorial: {forbidden}")

    code_text = "\n".join(source for _, source in parsed_code)
    require(re.search(r"\bDIMER_[A-Z0-9_]+\b", code_text) is None, "standalone tutorial must not depend on DIMER_* runtime variables")
    require(not repo_internal_imports(parsed_code), "standalone tutorial must not import repo-internal worker modules")

    for name, expected in (
        ("RUN_FINE_TUNING", False),
        ("RUN_NEW_DATA_INFERENCE", False),
        ("FINE_TUNE_STEPS", 50),
        ("MAX_MEMORY_USAGE_RATIO", 1.1),
        ("MIN_SELECTION_HOLDOUT_ROWS", 50),
        ("PROBLEM_TYPE", "regression"),
        ("NETWORK_TIMEOUT_SECONDS", 30),
    ):
        require(top_level_literal_assignments_match(parsed_code, name, expected), f"every top-level assignment to {name} must be literal {expected!r}")

    require(mitra_metric_map_is_regression(parsed_code), "MITRA_METRIC_MAP must map MAE/RMSE to Mitra native metric names")
    require(tabular_predictor_is_regression(parsed_code), "training tutorial must construct TabularPredictor with problem_type='regression'")
    require(metadata_records_regression(parsed_code), "export metadata must record problem_type='regression'")
    require(has_memory_guard(parsed_code), "tutorial must pass max_memory_usage_ratio through .fit(...)")
    require(has_current_fit_completion_gate(parsed_code), "current-fit completion gate missing")
    require(fit_completion_gate_uses(parsed_code) >= 2, "inference and export must both gate on the current fit")
    require(inference_step_is_self_contained(parsed_code), "Step 5 inference must import csv/io/pandas locally")
    require(has_safe_direct_weights_copy_guard(parsed_code), "direct model.safetensors upload must avoid copying a path onto itself")

    attrs = call_attributes(parsed_code)
    require("predict" in attrs, "training tutorial must call predict")
    require("predict_proba" not in attrs, "regression tutorial must not call predict_proba")


def validate_inference_tutorial():
    _, text, parsed_code = load_notebook(INFERENCE_NOTEBOOK)
    for marker in (
        "autogluon.tabular[mitra]==1.5.0",
        "mitra-predictor.zip",
        "predictor.pkl",
        "tutorial_run_metadata.json",
        "safe_extract_zip",
        "stat.S_IFLNK",
        "EXPECTED_ZIP_SHA256",
        "trusted source",
        "already contains a 'prediction' column",
        "TabularPredictor.load",
        "problem_type != 'regression'",
        "FEATURE_COLUMNS",
        "Inference CSV contains duplicate column names",
        "prediction",
        "predictions.csv",
        "GPT-5.6 Sol High",
        "OpenAI / ChatGPT",
        "Agent Relay role",
        "provenance, not sign-off",
    ):
        require(marker in text, f"inference tutorial missing required marker: {marker}")

    code_text = "\n".join(source for _, source in parsed_code)
    require(re.search(r"\bDIMER_[A-Z0-9_]+\b", code_text) is None, "inference tutorial must not depend on DIMER_* runtime variables")
    for forbidden in ("model.safetensors", "config.json", "hf_hub_download", "huggingface_hub", "fine_tune_steps", "predict_proba"):
        require(forbidden not in code_text, f"inference tutorial must not reacquire/train/classify: {forbidden}")

    require(not repo_internal_imports(parsed_code), "inference tutorial must not import worker modules")
    attrs = call_attributes(parsed_code)
    require("fit" not in attrs, "inference tutorial must not call fit")
    require("load" in attrs and "predict" in attrs, "inference tutorial must load and predict")
    require(not direct_tabular_predictor_construction(parsed_code), "inference tutorial must reload, not construct, a TabularPredictor")


def validate_docs():
    text = TUTORIAL_README.read_text(encoding="utf-8")
    for marker in (
        PINNED_REVISION, WEIGHTS_SHA256, CONFIG_SHA256, SAMPLE_REVISION,
        "freshretailnet-h7.zip", "purged per-series chronological split",
        "CC BY 4.0", "mitra_regressor_predictor_inference_colab.ipynb",
        "mean_absolute_error", "root_mean_squared_error",
        "TabularPredictor.load", "predictions.csv",
        "GPT-5.6 Sol High", "OpenAI / ChatGPT", "Agent Relay role",
        "provenance, not sign-off",
    ):
        require(marker in text, f"tutorial README missing marker: {marker}")


def main():
    validate_training_tutorial()
    validate_inference_tutorial()
    validate_docs()
    print("Standalone Mitra Regressor Colab tutorials: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
