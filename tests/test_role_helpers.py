"""Offline tests for the fleet snapshot helpers, the serving wrapper and the role-stage helpers.

No weights, no model: snapshot directories are temporary, the downloader is injected, the digest constants are
redirected to stand-in files where the happy path needs package constants and manifest to agree, and AutoGluon
is never imported.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from mitra_pipeline import (
    CONFIG_FILE,
    INPUT_SCHEMA,
    MANIFEST_NAME,
    MAX_FEATURES,
    MAX_TRAIN_ROWS,
    METRIC_IDS,
    MIN_TRAIN_ROWS,
    MODEL_ID,
    MODEL_KEY,
    MODEL_LICENSE,
    MODEL_REVISION,
    PINNED_REVISION,
    WEIGHTS_FILE,
    MitraRegressionPipeline,
    evaluation_report,
    regression_metrics,
    stage_missing_files,
    training_mean_baseline,
    validate_inference_frame,
    validate_inputs,
    validate_labeled_frame,
    verify_snapshot,
)
from mitra_pipeline import tutorial_api as api

WEIGHTS_PAYLOAD = b"not-the-real-safetensors"
CONFIG_PAYLOAD = b'{"dim": 1}'
WDIGEST = hashlib.sha256(WEIGHTS_PAYLOAD).hexdigest()
CDIGEST = hashlib.sha256(CONFIG_PAYLOAD).hexdigest()


def _manifest(wsha: str = WDIGEST, csha: str = CDIGEST) -> dict:
    return {
        "format": "dimer_hf_snapshot",
        "formatVersion": 1,
        "modelKey": MODEL_KEY,
        "modelId": MODEL_ID,
        "revision": MODEL_REVISION,
        "files": [
            {"path": WEIGHTS_FILE, "bytes": len(WEIGHTS_PAYLOAD), "sha256": wsha},
            {"path": CONFIG_FILE, "bytes": len(CONFIG_PAYLOAD), "sha256": csha},
        ],
        "totalBytes": len(WEIGHTS_PAYLOAD) + len(CONFIG_PAYLOAD),
    }


def _snapshot(tmp_path: Path, *, manifest: dict | None = None, write_files: bool = True) -> Path:
    root = tmp_path / "weights" / MODEL_KEY
    root.mkdir(parents=True)
    (root / MANIFEST_NAME).write_text(json.dumps(manifest or _manifest()), encoding="utf-8")
    if write_files:
        (root / WEIGHTS_FILE).write_bytes(WEIGHTS_PAYLOAD)
        (root / CONFIG_FILE).write_bytes(CONFIG_PAYLOAD)
    return root


@pytest.fixture
def constants(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(api, "WEIGHTS_SHA256", WDIGEST)
    monkeypatch.setattr(api, "CONFIG_SHA256", CDIGEST)


def test_identity_constants_are_the_fleet_shape() -> None:
    assert MODEL_ID == "autogluon/mitra-regressor"
    assert MODEL_REVISION == PINNED_REVISION
    assert len(MODEL_REVISION) == 40 and all(c in "0123456789abcdef" for c in MODEL_REVISION)
    assert MODEL_LICENSE == "apache-2.0"
    assert api.DEFAULT_WEIGHTS_DIR.name == MODEL_KEY and api.DEFAULT_WEIGHTS_DIR.parent.name == "weights"
    assert api.DEFAULT_WEIGHTS_DIR.parent.parent == Path(api.__file__).resolve().parents[1]  # root-level package


def test_committed_manifest_matches_the_digest_constants() -> None:
    manifest_path = api.DEFAULT_WEIGHTS_DIR / MANIFEST_NAME
    if not manifest_path.is_file():
        pytest.skip("snapshot manifest is not staged in this checkout")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert (manifest["modelId"], manifest["revision"]) == (MODEL_ID, MODEL_REVISION)
    digests = {entry["path"]: entry["sha256"] for entry in manifest["files"]}
    assert digests[WEIGHTS_FILE] == api.WEIGHTS_SHA256 and digests[CONFIG_FILE] == api.CONFIG_SHA256


def test_verify_snapshot_happy_path_and_rejections(tmp_path: Path, constants: None) -> None:
    root = _snapshot(tmp_path)
    result = verify_snapshot(root)
    assert result["path"] == str(root) and [f["path"] for f in result["files"]] == [WEIGHTS_FILE, CONFIG_FILE]
    (root / WEIGHTS_FILE).write_bytes(WEIGHTS_PAYLOAD[:-1] + b"!")  # same size, different bytes
    with pytest.raises(ValueError, match="sha256"):
        verify_snapshot(root)
    with pytest.raises(ValueError, match="package constant"):
        verify_snapshot(_snapshot(tmp_path / "bad", manifest=_manifest(wsha="0" * 64)))
    with pytest.raises(ValueError, match="modelId"):
        verify_snapshot(_snapshot(tmp_path / "foreign", manifest={**_manifest(), "modelId": "x/y"}))
    with pytest.raises(FileNotFoundError, match="snapshot file missing"):
        verify_snapshot(_snapshot(tmp_path / "missing", write_files=False))


def test_stage_missing_files_contract(tmp_path: Path, constants: None) -> None:
    assert stage_missing_files(_snapshot(tmp_path)) == []
    root = _snapshot(tmp_path / "empty", write_files=False)
    with pytest.raises(FileNotFoundError, match="allow_download=True"):
        stage_missing_files(root)
    calls: list[tuple[str, Path]] = []

    def downloader(relative_path: str, destination: Path) -> None:
        calls.append((relative_path, destination))
        payload = WEIGHTS_PAYLOAD if relative_path == WEIGHTS_FILE else CONFIG_PAYLOAD
        (destination / relative_path).write_bytes(payload)

    assert stage_missing_files(root, allow_download=True, downloader=downloader) == [WEIGHTS_FILE, CONFIG_FILE]
    assert [c[0] for c in calls] == [WEIGHTS_FILE, CONFIG_FILE]
    with pytest.raises(ValueError, match="refusing to stage"):
        stage_missing_files(_snapshot(tmp_path / "rev", manifest={**_manifest(), "revision": "a" * 40}, write_files=False), allow_download=True, downloader=lambda *a: None)


def test_from_pretrained_verifies_then_stages_the_offline_hf_snapshot(tmp_path: Path, constants: None) -> None:
    root = _snapshot(tmp_path)
    staged: list[tuple[Path, Path, Path]] = []

    def fake_stage(weights_path, config_path, *, hf_home):
        staged.append((Path(weights_path), Path(config_path), Path(hf_home)))
        return Path(hf_home) / "snapshot"

    api_stage = api.stage_verified_hf_snapshot
    api.stage_verified_hf_snapshot = fake_stage
    try:
        pipe = MitraRegressionPipeline.from_pretrained(weights_dir=root, device="cpu")
    finally:
        api.stage_verified_hf_snapshot = api_stage
    assert staged == [(root / WEIGHTS_FILE, root / CONFIG_FILE, root / ".cache" / "hf")]
    assert pipe.model_weight_path == root / WEIGHTS_FILE and pipe.source == "local-snapshot" and pipe.device == "cpu"
    assert pipe.predictor is None
    with pytest.raises(RuntimeError, match="not fitted"):
        pipe.predict(pd.DataFrame({"x": [1.0]}))
    with pytest.raises(FileNotFoundError):
        MitraRegressionPipeline.from_pretrained(weights_dir=_snapshot(tmp_path / "u", write_files=False), device="cpu")


def _table(rows: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame({"x1": rng.normal(size=rows), "city": rng.choice(["a", "b"], size=rows), "target": rng.normal(size=rows)})


def test_validate_inputs_fit_mode_matches_validate_labeled_frame() -> None:
    frame = _table()
    frame.loc[0, "target"] = np.nan
    manifest = validate_inputs(frame, target_column="target", names=["sample"])
    assert manifest["verdict"] == "accepted" and manifest["findings"] == []
    assert manifest["schema"] == INPUT_SCHEMA
    assert manifest["schema"]["train_rows"] == [MIN_TRAIN_ROWS, MAX_TRAIN_ROWS]
    assert manifest["schema"]["features"] == [1, MAX_FEATURES]
    (entry,) = manifest["inputs"]
    assert entry["id"] == "sample" and entry["mode"] == "fit" and entry["rows"] == 59
    assert entry["feature_columns"] == ["x1", "city"] and entry["categorical_columns"] == ["city"]
    assert entry["report"]["dropped_missing_target_rows"] == 1
    _clean, _features, report = validate_labeled_frame(frame, "target", name="sample", min_rows=MIN_TRAIN_ROWS, require_variation=True)
    assert entry["report"] == report
    assert (manifest["model_id"], manifest["model_revision"]) == (MODEL_ID, MODEL_REVISION)
    for bad, message in (
        (_table().rename(columns={"target": "y"}), "target 'target' not found"),
        (_table(10), "at least 50 labelled rows"),
        (_table().assign(target="x"), "target must be numeric"),
        (_table().assign(target=1.0), "no variation"),
    ):
        with pytest.raises(ValueError, match=message):
            validate_inputs(bad, target_column="target")
        with pytest.raises(ValueError, match=message):
            validate_labeled_frame(bad, "target", name="table-0", min_rows=MIN_TRAIN_ROWS, require_variation=True)
    with pytest.raises(ValueError, match="names must have exactly one entry"):
        validate_inputs(_table(), names=["a", "b"])


def test_validate_inputs_inference_mode_matches_validate_inference_frame() -> None:
    rows = _table(3).drop(columns=["target"]).assign(extra=1)
    manifest = validate_inputs(rows, None, feature_columns=["x1", "city"], names=["new"])
    (entry,) = manifest["inputs"]
    assert entry == {"id": "new", "mode": "inference", "rows": 3, "feature_columns": ["x1", "city"], "extra_columns": ["extra"], "missing_value_columns": {}}
    for bad, message in (
        (rows.drop(columns=["city"]), "missing required feature columns"),
        (rows.assign(prediction=1.0), "already contains 'prediction'"),
    ):
        with pytest.raises(ValueError, match=message):
            validate_inputs(bad, None, feature_columns=["x1", "city"])
        with pytest.raises(ValueError, match=message):
            validate_inference_frame(bad, ["x1", "city"])
    with pytest.raises(ValueError, match="feature_columns is required"):
        validate_inputs(rows, None)


def test_training_mean_baseline_and_evaluation_report() -> None:
    baseline = training_mean_baseline([1.0, 2.0, 3.0], [2.0, 4.0])
    assert set(baseline) == set(METRIC_IDS) and baseline["mae"] == 1.0 and baseline["r2"] == pytest.approx(-1.0)
    with pytest.raises(ValueError, match="non-empty and finite"):
        training_mean_baseline([1.0, np.inf], [1.0])
    metrics = regression_metrics([1.0, 2.0, 3.0, 4.0], [1.1, 2.1, 2.9, 4.2])
    report = evaluation_report(metrics, baseline=baseline, independent_test=metrics, n_holdout=4, n_test=4, target_column="target", selection="default:pretrained", sample_kind="sample", estimation="e")
    assert report["verdict"] == "sample-sanity"
    assert [m["id"] for m in report["metrics"]] == list(METRIC_IDS)
    assert all(m["estimation"] == "e" for m in report["metrics"])
    assert [m["id"] for m in report["independent_test"]] == list(METRIC_IDS)
    assert report["baselines"][0]["id"] == "training_mean" and report["selection"] == "default:pretrained"
    nan = evaluation_report({"mae": 1.0, "r2": float("nan")}, n_holdout=1)
    assert nan["metrics"][1]["value"] is None
    missing = evaluation_report(None, n_holdout=0, target_column="target", sample_kind="BYOD")
    assert missing["verdict"] == "not-measurable" and "training_mean_baseline" in missing["needs"]
    assert (missing["model_id"], missing["model_revision"]) == (MODEL_ID, MODEL_REVISION)
    with pytest.raises(ValueError, match="unknown metric ids"):
        evaluation_report({"mape": 0.5})
