#!/usr/bin/env python3
"""Static Notebook Specification 1.0 checks for Mitra Regressor tutorials."""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "tutorials" / "mitra_regressor_colab.ipynb"
INFERENCE = ROOT / "tutorials" / "mitra_regressor_predictor_inference_colab.ipynb"
TUTORIAL_README = ROOT / "tutorials" / "README.md"
PUBLIC_API = ROOT / "mitra_pipeline" / "tutorial_api.py"
REQUIREMENTS = ROOT / "tutorials" / "requirements-colab.txt"

MODEL_ID = "autogluon/mitra-regressor"
PINNED_REVISION = "5f277aa8f69042d39d6ac3612aed18bb9279bd95"
WEIGHTS_SHA256 = "d8e75c62af0bec2fd404b0ad20a442d951d43ca6d331315cfcc0509b54f2c642"
CONFIG_SHA256 = "2bc1ed5047f7c25368245e8ad32540a5fa28940b1ec05d3f1f454a09ff5384c1"
SAMPLE_REVISION = "f02e0c38ce835d6b85b5a6f072d232f3cd306f54"
PLACEHOLDERS = re.compile(r"\b(TODO|TBD|FIXME)\b")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def source_text(cell: dict) -> str:
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else str(source)


def load_notebook(path: Path) -> tuple[dict, str, list[str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    require(payload.get("nbformat") == 4, f"{path.name}: nbformat must be 4")
    cells = payload.get("cells", [])
    require(bool(cells), f"{path.name}: no cells")
    cell_ids = [cell.get("id") for cell in cells]
    require(
        all(isinstance(cell_id, str) and re.fullmatch(r"[A-Za-z0-9_-]{1,64}", cell_id) for cell_id in cell_ids),
        f"{path.name}: every cell must have a valid nbformat cell id",
    )
    require(len(set(cell_ids)) == len(cell_ids), f"{path.name}: cell ids must be unique")
    full_text = "\n".join(source_text(cell) for cell in cells)

    code_cells: list[str] = []
    for index, cell in enumerate(cells):
        if cell.get("cell_type") != "code":
            continue
        source = source_text(cell)
        stripped = "\n".join(
            line
            for line in source.splitlines()
            if not line.lstrip().startswith(("%", "!"))
        )
        if stripped.strip():
            ast.parse(stripped, filename=f"{path.name}:cell-{index}")
        code_cells.append(stripped)

    require(not PLACEHOLDERS.search(full_text), f"{path.name}: placeholder marker survives")
    require(all(cell.get("execution_count") is None for cell in cells if cell.get("cell_type") == "code"),
            f"{path.name}: execution counts must be cleared")
    require(all(not cell.get("outputs") for cell in cells if cell.get("cell_type") == "code"),
            f"{path.name}: persisted outputs must be cleared")
    return payload, full_text, code_cells


def require_profile(payload: dict, filename: str, expected: str) -> None:
    dimer = payload.get("metadata", {}).get("dimer", {})
    require(dimer.get("notebook_profile") == expected, f"{filename}: metadata profile must be {expected}")
    require(str(dimer.get("notebook_spec")) == "1.0", f"{filename}: notebook spec must be 1.0")


def require_markers(text: str, markers: tuple[str, ...], label: str) -> None:
    for marker in markers:
        require(marker in text, f"{label}: missing marker {marker!r}")


def top_level_literal(code_cells: list[str], name: str, expected) -> bool:
    found = False
    for source in code_cells:
        if not source.strip():
            continue
        tree = ast.parse(source)
        for node in tree.body:
            if not isinstance(node, ast.Assign):
                continue
            if not any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
                continue
            found = True
            if not isinstance(node.value, ast.Constant) or node.value.value != expected:
                return False
    return found


def called_attribute(code_cells: list[str], name: str) -> bool:
    for source in code_cells:
        if not source.strip():
            continue
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == name:
                return True
    return False


def main_tutorial() -> None:
    payload, text, code_cells = load_notebook(MAIN)
    require_profile(payload, MAIN.name, "E2E")
    require_markers(
        text,
        (
            "**Profile:** `E2E`",
            "**Notebook specification:** `1.0`",
            "mitra_pipeline",
            "fit_mitra_predictor",
            "validate_labeled_frame",
            "split_overlap_report",
            "DummyRegressor",
            "mean",
            "median",
            "mean_absolute_error",
            "root_mean_squared_error",
            "point estimate",
            "no calibrated per-prediction interval",
            "artifact_manifest.json",
            "tutorial_run_metadata.json",
            "write_artifact_manifest",
            "validate_artifact_directory",
            "safe_extract_archive",
            "np.allclose",
            "dimer-model-manifest.json",
            "tutorial_metrics.json",
            "SAMPLE_TRAIN_ROWS",
            "SAMPLE_EVAL_ROWS",
            "Default smoke subset:",
            "If every default-path cell ran successfully",
            "It **does not** establish",
        ),
        MAIN.name,
    )
    require(top_level_literal(code_cells, "SAMPLE_REVISION", SAMPLE_REVISION),
            f"main notebook: SAMPLE_REVISION must match certified revision {SAMPLE_REVISION}")
    require(top_level_literal(code_cells, "SAMPLE_TRAIN_ROWS", 512),
            "main notebook: SAMPLE_TRAIN_ROWS must default 512")
    require(top_level_literal(code_cells, "SAMPLE_EVAL_ROWS", 256),
            "main notebook: SAMPLE_EVAL_ROWS must default 256")
    require(top_level_literal(code_cells, "RUN_FINE_TUNING", False),
            "main notebook: RUN_FINE_TUNING must default False")
    require(top_level_literal(code_cells, "RUN_NEW_DATA_INFERENCE", False),
            "main notebook: RUN_NEW_DATA_INFERENCE must default False")
    require("mp.predict_regression" in "\n".join(code_cells), "main notebook must exercise repository prediction API")
    require("predict_proba" not in text, "main regression notebook must not call predict_proba")
    require("lightgbm>=4.0,<4.8" not in text, "main notebook must not use floating LightGBM range")
    require("requirements-colab.txt" in text, "main notebook must install pinned tutorial requirements")
    require("DIMER_EXPECTED_ZIP_SHA256" in text, "main notebook must support non-interactive DIMER ZIP digest input")


def inference_tutorial() -> None:
    payload, text, code_cells = load_notebook(INFERENCE)
    require_profile(payload, INFERENCE.name, "ARTIFACT-INFERENCE")
    require_markers(
        text,
        (
            "**Profile:** `ARTIFACT-INFERENCE`",
            "**Notebook specification:** `1.0`",
            "MITRA_PREDICTOR_ZIP",
            "MITRA_INFERENCE_CSV",
            "MITRA_EXPECTED_ZIP_SHA256",
            "ALLOW_UNVERIFIED_ARTIFACT",
            "artifact_manifest.json",
            "tutorial_run_metadata.json",
            "validate_artifact_directory",
            "safe_extract_archive",
            "TabularPredictor.load",
            "point estimate",
            "No per-prediction uncertainty interval",
            "predictions.csv",
            "A successful run proves",
            "It does **not** prove",
        ),
        INFERENCE.name,
    )
    require(top_level_literal(code_cells, "ALLOW_UNVERIFIED_ARTIFACT", False),
            "artifact-inference notebook must fail closed unless unverified loading is explicitly enabled")
    require("fit(" not in "\n".join(code_cells), "artifact-inference notebook must not fit/train")
    require("model.safetensors" not in "\n".join(code_cells),
            "artifact-inference notebook must not reacquire base weights")
    require(called_attribute(code_cells, "load"), "artifact-inference notebook must load predictor state")
    require(called_attribute(code_cells, "to_csv"), "artifact-inference notebook must export CSV")


def docs_and_api() -> None:
    readme = TUTORIAL_README.read_text(encoding="utf-8")
    api = PUBLIC_API.read_text(encoding="utf-8")
    req = REQUIREMENTS.read_text(encoding="utf-8")

    require_markers(
        readme,
        (
            "DIMER Notebook Specification:** `1.0`",
            "`E2E`",
            "`ARTIFACT-INFERENCE`",
            "release-grade candidate",
            "Notebook release execution",
            MODEL_ID,
            PINNED_REVISION,
            WEIGHTS_SHA256,
            CONFIG_SHA256,
            SAMPLE_REVISION,
            "512 training rows",
            "256 rows from each evaluation partition",
        ),
        "tutorials/README.md",
    )
    require("autogluon.tabular[mitra]==1.5.0" in req, "tutorial requirements must pin AutoGluon")
    require("lightgbm==4.6.0" in req, "tutorial requirements must pin LightGBM")
    require(">=" not in req and "<" not in req, "tutorial direct requirements must not float")

    require_markers(
        api,
        (
            f'MODEL_ID = "{MODEL_ID}"',
            f'PINNED_REVISION = "{PINNED_REVISION}"',
            f'WEIGHTS_SHA256 = "{WEIGHTS_SHA256}"',
            f'CONFIG_SHA256 = "{CONFIG_SHA256}"',
            'ARTIFACT_FORMAT = "dimer-autogluon-predictor"',
            "ARTIFACT_FORMAT_VERSION = 1",
            "MAX_ARCHIVE_MEMBER_BYTES",
            "MAX_ARCHIVE_EXPANDED_BYTES",
            "MAX_COMPRESSION_RATIO",
            "validate_dimer_model_package",
            "write_artifact_manifest",
            "validate_artifact_directory",
            "safe_extract_archive",
            "fit_mitra_predictor",
        ),
        "mitra_pipeline/tutorial_api.py",
    )


def main() -> int:
    main_tutorial()
    inference_tutorial()
    docs_and_api()
    print("Mitra Regressor Notebook Specification 1.0 static conformance: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
