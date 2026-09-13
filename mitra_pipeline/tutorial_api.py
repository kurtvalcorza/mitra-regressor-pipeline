"""Release-grade public API for the Mitra Regressor notebook surface.

This module owns the repository-facing behavior the notebooks must exercise directly:
regression data validation, Mitra fit/predict calls, model snapshot locking, and predictor
artifact validation. Imports that pull in heavy ML dependencies are intentionally delayed so
archive and manifest checks remain unit-testable in a lightweight CI environment.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import random
import shutil
import stat
import tempfile
import urllib.request
import zipfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import pandas as pd

# Fleet snapshot identity (DIMER Notebook Specification 1.1, ST3/MOD13). The pinned upstream model is unchanged;
# these are the fleet-standard names for the same repository, revision, license and snapshot key. The published
# PINNED_REVISION spelling stays as an alias of MODEL_REVISION.
MODEL_ID = "autogluon/mitra-regressor"
MODEL_REVISION = "5f277aa8f69042d39d6ac3612aed18bb9279bd95"
MODEL_LICENSE = "apache-2.0"
MODEL_KEY = "mitra-regressor"
MANIFEST_NAME = "dimer-base-manifest.json"
# Root-level package: the repository root is one level up from this file.
DEFAULT_WEIGHTS_DIR = Path(__file__).resolve().parents[1] / "weights" / MODEL_KEY
WEIGHTS_FILE = "model.safetensors"
CONFIG_FILE = "config.json"

PINNED_REVISION = MODEL_REVISION
WEIGHTS_SHA256 = "d8e75c62af0bec2fd404b0ad20a442d951d43ca6d331315cfcc0509b54f2c642"
CONFIG_SHA256 = "2bc1ed5047f7c25368245e8ad32540a5fa28940b1ec05d3f1f454a09ff5384c1"
METRIC_IDS = ("mae", "rmse", "r2")  # the ids `regression_metrics` reports

MAX_TRAIN_ROWS = 10_000
MAX_FEATURES = 500
MIN_TRAIN_ROWS = 50

ARTIFACT_FORMAT = "dimer-autogluon-predictor"
ARTIFACT_FORMAT_VERSION = 1
ARTIFACT_MANIFEST = "artifact_manifest.json"
RUN_METADATA = "tutorial_run_metadata.json"
DIMER_MODEL_MANIFEST = "dimer-model-manifest.json"

MAX_ARCHIVE_MEMBER_BYTES = 1 * 1024**3
MAX_ARCHIVE_EXPANDED_BYTES = 4 * 1024**3
MAX_COMPRESSION_RATIO = 200.0

REGRESSION_LOWER_IS_BETTER = {
    "root_mean_squared_error",
    "mean_squared_error",
    "mean_absolute_error",
    "median_absolute_error",
    "mean_absolute_percentage_error",
    "symmetric_mean_absolute_percentage_error",
    "root_mean_squared_logarithmic_error",
}
MITRA_METRIC_MAP = {
    "mean_absolute_error": "mae",
    "root_mean_squared_error": "rmse",
}

REQUIRED_RUN_METADATA_FIELDS = {
    "artifact_format",
    "artifact_format_version",
    "base_model",
    "base_model_revision",
    "weights_sha256",
    "config_sha256",
    "autogluon_version",
    "python_version",
    "problem_type",
    "target_column",
    "features",
    "mode",
    "selection_basis",
}


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _assert_hex_digest(value: str, label: str) -> str:
    digest = str(value).strip().lower()
    if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
        raise ValueError(f"{label} must be a 64-character hexadecimal SHA-256 digest.")
    return digest


def read_csv_bytes(payload: bytes, label: str) -> pd.DataFrame:
    """Parse UTF-8/BOM CSV bytes while rejecting duplicate raw headers before pandas renames them."""
    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{label}: expected UTF-8 CSV input.") from exc

    rows = csv.reader(io.StringIO(text, newline=""))
    header = next((row for row in rows if row and not (len(row) == 1 and not row[0].strip())), [])
    if not header:
        raise ValueError(f"{label}: CSV has no header row.")

    seen: set[str] = set()
    duplicates: list[str] = []
    for name in header:
        if name in seen and name not in duplicates:
            duplicates.append(name)
        seen.add(name)
    if duplicates:
        raise ValueError(f"{label}: duplicate column names are not supported: {duplicates}")

    return pd.read_csv(io.BytesIO(payload))


def validate_labeled_frame(
    frame: pd.DataFrame,
    target_column: str,
    *,
    name: str,
    drop_columns: Iterable[str] = (),
    min_rows: int = 2,
    require_variation: bool = False,
) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    """Validate one labelled regression table using the repository's user-facing contract."""
    if frame.columns.duplicated().any():
        duplicates = list(frame.columns[frame.columns.duplicated()])
        raise ValueError(f"{name}: duplicate column names are not supported: {duplicates}")
    if target_column not in frame.columns:
        raise ValueError(f"{name}: target {target_column!r} not found.")

    drops = [c for c in drop_columns if c != target_column and c in frame.columns]
    out = frame.drop(columns=drops, errors="ignore").copy()
    raw_target = out[target_column]
    numeric_target = pd.to_numeric(raw_target, errors="coerce")
    non_numeric = raw_target.notna() & numeric_target.isna()
    if bool(non_numeric.any()):
        examples = raw_target[non_numeric].astype(str).head(5).tolist()
        raise ValueError(f"{name}: target must be numeric; invalid examples: {examples}")
    out[target_column] = numeric_target

    rows_before = len(out)
    out = out.dropna(subset=[target_column]).copy()
    dropped_target_rows = rows_before - len(out)
    target_values = out[target_column].to_numpy(dtype=float)
    if not np.isfinite(target_values).all():
        raise ValueError(f"{name}: target contains infinite values; use finite numeric regression targets only.")

    features = [c for c in out.columns if c != target_column]
    errors: list[str] = []
    if len(out) < min_rows:
        errors.append(f"use at least {min_rows} labelled rows after missing-target drops")
    if not features:
        errors.append("no feature columns remain")
    if len(features) > MAX_FEATURES:
        errors.append(f"{len(features)} features exceed the {MAX_FEATURES}-feature Mitra ceiling")
    if require_variation and out[target_column].nunique(dropna=True) < 2:
        errors.append("training target has no variation")
    if errors:
        raise ValueError(f"{name} is not ready: " + "; ".join(errors))

    report = {
        "rows_before_target_drop": int(rows_before),
        "rows_after_target_drop": int(len(out)),
        "dropped_missing_target_rows": int(dropped_target_rows),
        "exact_duplicate_rows": int(out.duplicated().sum()),
        "feature_count": int(len(features)),
    }
    return out, features, report


def cap_training_rows(
    frame: pd.DataFrame,
    target_column: str,
    *,
    seed: int,
    max_rows: int = MAX_TRAIN_ROWS,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if max_rows > MAX_TRAIN_ROWS:
        raise ValueError(f"max_rows cannot exceed Mitra's {MAX_TRAIN_ROWS:,}-row ceiling.")
    before = len(frame)
    if before <= max_rows:
        return frame.copy(), {"applied": False, "before": before, "after": before}
    capped = frame.sample(n=max_rows, random_state=seed).copy()
    if capped[target_column].nunique(dropna=True) < 2:
        raise ValueError("Capped training split has no target variation; provide a representative pre-split set.")
    return capped, {"applied": True, "before": before, "after": len(capped)}


def split_overlap_report(named_frames: dict[str, pd.DataFrame]) -> dict[str, int]:
    """Detect exact record overlap across labelled partitions without mutating them."""
    names = list(named_frames)
    hashes = {
        name: set(pd.util.hash_pandas_object(frame, index=False).astype("uint64").tolist())
        for name, frame in named_frames.items()
    }
    report: dict[str, int] = {}
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            report[f"{left}_vs_{right}"] = len(hashes[left].intersection(hashes[right]))
    return report


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def fit_mitra_predictor(
    train_data: pd.DataFrame,
    *,
    target_column: str,
    eval_metric: str,
    path: str | Path,
    fine_tune: bool,
    time_limit: int,
    seed: int,
    fine_tune_steps: int | None = None,
    max_memory_usage_ratio: float = 1.10,
):
    """Fit/register Mitra through the repository's supported notebook API.

    With ``fine_tune=False`` AutoGluon's ``fit`` registers support/context rows and model
    configuration; Mitra weights are not gradient-updated. With ``fine_tune=True`` the Mitra
    weights are adapted for the requested number of steps, subject to the time limit.
    """
    if eval_metric not in MITRA_METRIC_MAP:
        raise ValueError(f"Unsupported eval_metric {eval_metric!r}; choose {sorted(MITRA_METRIC_MAP)}")
    if fine_tune and (fine_tune_steps is None or fine_tune_steps <= 0):
        raise ValueError("fine_tune_steps must be a positive integer when fine_tune=True.")

    seed_everything(seed)
    hp: dict[str, Any] = {
        "fine_tune": bool(fine_tune),
        "seed": int(seed),
        "metric": MITRA_METRIC_MAP[eval_metric],
    }
    if fine_tune:
        hp["fine_tune_steps"] = int(fine_tune_steps)

    from autogluon.tabular import TabularPredictor

    predictor = TabularPredictor(
        label=target_column,
        problem_type="regression",
        eval_metric=eval_metric,
        path=str(path),
        verbosity=2,
    )
    predictor.fit(
        train_data,
        hyperparameters={"MITRA": hp},
        fit_weighted_ensemble=False,
        time_limit=int(time_limit),
        ag_args_fit={"max_memory_usage_ratio": float(max_memory_usage_ratio)},
    )
    trained = list(predictor.model_names())
    if not trained or not any("mitra" in model.lower() for model in trained):
        raise RuntimeError(f"Expected Mitra to fit/register; AutoGluon returned models={trained}.")
    return predictor


def normalize_autogluon_regression_metrics(raw: dict[str, Any]) -> dict[str, float]:
    return {
        key: float(-value if key in REGRESSION_LOWER_IS_BETTER else value)
        for key, value in raw.items()
    }


def regression_metrics(y_true: Iterable[float], y_pred: Iterable[float]) -> dict[str, float]:
    truth = np.asarray(list(y_true), dtype=float)
    pred = np.asarray(list(y_pred), dtype=float)
    if truth.shape != pred.shape:
        raise ValueError(f"y_true shape {truth.shape} != y_pred shape {pred.shape}")
    err = truth - pred
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    denom = float(np.sum((truth - np.mean(truth)) ** 2))
    r2 = float(1.0 - np.sum(err**2) / denom) if denom > 0 else float("nan")
    return {"mae": mae, "rmse": rmse, "r2": r2}


def validate_inference_frame(
    frame: pd.DataFrame,
    required_features: Iterable[str],
    *,
    output_column: str = "prediction",
) -> tuple[pd.DataFrame, list[str]]:
    if frame.columns.duplicated().any():
        duplicates = list(frame.columns[frame.columns.duplicated()])
        raise ValueError(f"Inference CSV contains duplicate column names: {duplicates}")
    required = list(required_features)
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Inference CSV is missing required feature columns: {missing}")
    if output_column in frame.columns:
        raise ValueError(f"Inference CSV already contains {output_column!r}; rename or remove it first.")
    extra = [column for column in frame.columns if column not in required]
    return frame.reindex(columns=required).copy(), extra


def predict_regression(predictor, frame: pd.DataFrame, required_features: Iterable[str]) -> np.ndarray:
    features = list(required_features)
    missing = [column for column in features if column not in frame.columns]
    if missing:
        raise ValueError(f"Prediction frame is missing required features: {missing}")
    predictions = predictor.predict(frame.reindex(columns=features))
    return np.asarray(predictions, dtype=float)


def _download(url: str, destination: Path, *, timeout: int = 30) -> None:
    with urllib.request.urlopen(url, timeout=timeout) as response, open(destination, "wb") as handle:
        shutil.copyfileobj(response, handle)


def stage_verified_hf_snapshot(
    weights_path: str | Path,
    config_path: str | Path,
    *,
    hf_home: str | Path,
) -> Path:
    """Verify exact Mitra bytes and stage them as the immutable pinned Hugging Face snapshot."""
    weights = Path(weights_path)
    config = Path(config_path)
    if sha256_file(weights) != WEIGHTS_SHA256:
        raise RuntimeError("model.safetensors checksum mismatch for the pinned Mitra Regressor release.")
    if sha256_file(config) != CONFIG_SHA256:
        raise RuntimeError("config.json checksum mismatch for the pinned Mitra Regressor release.")

    home = Path(hf_home)
    repo = home / "hub" / ("models--" + MODEL_ID.replace("/", "--"))
    snapshot = repo / "snapshots" / PINNED_REVISION
    refs = repo / "refs"
    snapshot.mkdir(parents=True, exist_ok=True)
    refs.mkdir(parents=True, exist_ok=True)
    shutil.copy2(weights, snapshot / "model.safetensors")
    shutil.copy2(config, snapshot / "config.json")
    (refs / "main").write_text(PINNED_REVISION, encoding="utf-8")

    os.environ["HF_HOME"] = str(home)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    from huggingface_hub import hf_hub_download

    for filename, expected_digest in (
        ("model.safetensors", WEIGHTS_SHA256),
        ("config.json", CONFIG_SHA256),
    ):
        resolved = Path(
            hf_hub_download(
                repo_id=MODEL_ID,
                filename=filename,
                revision=PINNED_REVISION,
                local_files_only=True,
            )
        ).resolve()
        expected = (snapshot / filename).resolve()
        if resolved != expected:
            raise RuntimeError(f"Offline resolver mismatch for {filename}: {resolved} != {expected}")
        if sha256_file(resolved) != expected_digest:
            raise RuntimeError(f"Resolved {filename} digest changed after staging.")
    return snapshot


def _validate_zip_members(zf: zipfile.ZipFile, destination: Path) -> list[zipfile.ZipInfo]:
    destination = destination.resolve()
    total = 0
    files: list[zipfile.ZipInfo] = []
    seen_paths: set[str] = set()
    seen_parent_paths: set[str] = set()
    for info in zf.infolist():
        name = info.filename
        if not name or info.is_dir():
            continue
        if "\\" in name:
            raise RuntimeError(f"Backslash archive paths are not allowed: {name!r}")
        member = PurePosixPath(name)
        if member.is_absolute() or ".." in member.parts:
            raise RuntimeError(f"Unsafe archive member path: {name!r}")
        normalized_name = member.as_posix()
        if normalized_name in seen_paths:
            raise RuntimeError(f"Duplicate archive member path is not allowed: {name!r}")
        parent_paths = {PurePosixPath(*member.parts[:i]).as_posix() for i in range(1, len(member.parts))}
        if normalized_name in seen_parent_paths or parent_paths.intersection(seen_paths):
            raise RuntimeError(f"Archive member path conflicts with a file/directory boundary: {name!r}")
        seen_paths.add(normalized_name)
        seen_parent_paths.update(parent_paths)
        mode = (info.external_attr >> 16) & 0o170000
        if mode == stat.S_IFLNK:
            raise RuntimeError(f"Symlink entries are not allowed: {name!r}")
        if info.file_size > MAX_ARCHIVE_MEMBER_BYTES:
            raise RuntimeError(f"Archive member is too large: {name!r} ({info.file_size:,} bytes).")
        if info.compress_size > 0 and info.file_size / info.compress_size > MAX_COMPRESSION_RATIO:
            raise RuntimeError(f"Archive member has suspicious compression ratio: {name!r}.")
        total += info.file_size
        if total > MAX_ARCHIVE_EXPANDED_BYTES:
            raise RuntimeError("Archive expanded size exceeds the configured safety ceiling.")
        target = (destination / Path(*member.parts)).resolve()
        if target != destination and destination not in target.parents:
            raise RuntimeError(f"Archive member escapes extraction root: {name!r}")
        files.append(info)
    return files


def safe_extract_archive(zip_path: str | Path, destination: str | Path) -> Path:
    destination_path = Path(destination)
    with zipfile.ZipFile(zip_path) as zf:
        # Validate the complete archive before touching any prior extraction destination.
        infos = _validate_zip_members(zf, destination_path)
        if destination_path.is_symlink():
            raise RuntimeError("Archive extraction destination must not be a symlink.")
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{destination_path.name}.extract-", dir=destination_path.parent))
        try:
            for info in infos:
                member = PurePosixPath(info.filename)
                target = staging.joinpath(*member.parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(info) as source, open(target, "wb") as sink:
                    shutil.copyfileobj(source, sink)
            if destination_path.exists():
                shutil.rmtree(destination_path)
            staging.replace(destination_path)
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise
    return destination_path


def validate_dimer_model_package(
    zip_path: str | Path,
    destination: str | Path,
    *,
    expected_archive_sha256: str | None = None,
) -> tuple[Path, Path, dict[str, Any]]:
    """Validate the normative DIMER offline Mitra package before any model load."""
    zip_path = Path(zip_path)
    if expected_archive_sha256:
        expected = _assert_hex_digest(expected_archive_sha256, "expected_archive_sha256")
        actual = sha256_file(zip_path)
        if actual != expected:
            raise RuntimeError(f"DIMER model ZIP checksum mismatch: expected {expected}; got {actual}.")

    root = safe_extract_archive(zip_path, destination)
    manifest_path = root / DIMER_MODEL_MANIFEST
    if not manifest_path.is_file():
        raise RuntimeError(
            f"DIMER model ZIP must contain root-level {DIMER_MODEL_MANIFEST}; legacy weight-only ZIPs do not satisfy Notebook Spec 1.0."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise RuntimeError("Unsupported DIMER model manifest schema_version.")
    if manifest.get("model_id") != MODEL_ID:
        raise RuntimeError(f"DIMER package model_id must be {MODEL_ID!r}.")
    if manifest.get("revision") != PINNED_REVISION:
        raise RuntimeError(f"DIMER package revision must be {PINNED_REVISION!r}.")

    declared = manifest.get("files")
    if not isinstance(declared, list) or not declared:
        raise RuntimeError("DIMER model manifest must contain a non-empty files list.")
    entries: dict[str, dict[str, Any]] = {}
    for entry in declared:
        if not isinstance(entry, dict):
            raise RuntimeError("DIMER model manifest file entries must be objects.")
        path = str(entry.get("path", ""))
        if not path or "\\" in path:
            raise RuntimeError(f"Invalid DIMER manifest path: {path!r}")
        member = PurePosixPath(path)
        if member.is_absolute() or ".." in member.parts or path in entries:
            raise RuntimeError(f"Unsafe or duplicate DIMER manifest path: {path!r}")
        entries[path] = entry

    for required in ("model.safetensors", "config.json"):
        if required not in entries:
            raise RuntimeError(f"DIMER model manifest is missing required file {required!r}.")

    actual_files = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path != root / DIMER_MODEL_MANIFEST
    }
    if actual_files != set(entries):
        missing = sorted(set(entries) - actual_files)
        unexpected = sorted(actual_files - set(entries))
        raise RuntimeError(f"DIMER model package file inventory mismatch; missing={missing}, unexpected={unexpected}")

    for rel, entry in entries.items():
        path = root / rel
        expected_size = int(entry.get("size", -1))
        expected_digest = _assert_hex_digest(str(entry.get("sha256", "")), f"sha256 for {rel}")
        if path.stat().st_size != expected_size:
            raise RuntimeError(f"DIMER model package size mismatch for {rel}.")
        if sha256_file(path) != expected_digest:
            raise RuntimeError(f"DIMER model package digest mismatch for {rel}.")

    weights = root / "model.safetensors"
    config = root / "config.json"
    if sha256_file(weights) != WEIGHTS_SHA256 or sha256_file(config) != CONFIG_SHA256:
        raise RuntimeError("DIMER package bytes do not match the pinned Mitra Regressor release.")
    return weights, config, manifest


def _artifact_inventory(root: Path) -> list[dict[str, Any]]:
    inventory = []
    for path in sorted(p for p in root.rglob("*") if p.is_file() and p != root / ARTIFACT_MANIFEST):
        rel = path.relative_to(root).as_posix()
        if "\\" in rel or PurePosixPath(rel).is_absolute() or ".." in PurePosixPath(rel).parts:
            raise RuntimeError(f"Unsafe artifact path: {rel!r}")
        inventory.append({"path": rel, "size": path.stat().st_size, "sha256": sha256_file(path)})
    return inventory


def write_artifact_manifest(root: str | Path) -> Path:
    root = Path(root)
    metadata_path = root / RUN_METADATA
    if not metadata_path.is_file():
        raise RuntimeError(f"Cannot manifest artifact without {RUN_METADATA}.")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    missing = sorted(REQUIRED_RUN_METADATA_FIELDS - set(metadata))
    if missing:
        raise RuntimeError(f"Run metadata missing required fields: {missing}")
    if metadata.get("artifact_format") != ARTIFACT_FORMAT:
        raise RuntimeError("Run metadata artifact_format mismatch.")
    if metadata.get("artifact_format_version") != ARTIFACT_FORMAT_VERSION:
        raise RuntimeError("Run metadata artifact_format_version mismatch.")

    manifest = {
        "schema_version": 1,
        "artifact_format": ARTIFACT_FORMAT,
        "artifact_format_version": ARTIFACT_FORMAT_VERSION,
        "base_model": metadata["base_model"],
        "base_model_revision": metadata["base_model_revision"],
        "problem_type": metadata["problem_type"],
        "metadata_file": RUN_METADATA,
        "files": _artifact_inventory(root),
    }
    path = root / ARTIFACT_MANIFEST
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return path


def validate_artifact_directory(
    root: str | Path,
    *,
    expected_model_id: str = MODEL_ID,
    expected_revision: str = PINNED_REVISION,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify format, provenance, inventory, sizes and digests before deserializing predictor state."""
    root = Path(root)
    manifest_path = root / ARTIFACT_MANIFEST
    metadata_path = root / RUN_METADATA
    if not manifest_path.is_file():
        raise RuntimeError(f"Predictor artifact is missing required {ARTIFACT_MANIFEST}.")
    if not metadata_path.is_file():
        raise RuntimeError(f"Predictor artifact is missing required {RUN_METADATA}.")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise RuntimeError("Unsupported artifact manifest schema_version.")
    if manifest.get("metadata_file") != RUN_METADATA:
        raise RuntimeError(f"Artifact manifest metadata_file must be {RUN_METADATA!r}.")
    if manifest.get("artifact_format") != ARTIFACT_FORMAT or metadata.get("artifact_format") != ARTIFACT_FORMAT:
        raise RuntimeError("Predictor artifact_format is not the DIMER AutoGluon predictor format.")
    if (
        manifest.get("artifact_format_version") != ARTIFACT_FORMAT_VERSION
        or metadata.get("artifact_format_version") != ARTIFACT_FORMAT_VERSION
    ):
        raise RuntimeError("Predictor artifact format version is incompatible with this notebook.")

    missing = sorted(REQUIRED_RUN_METADATA_FIELDS - set(metadata))
    if missing:
        raise RuntimeError(f"Predictor provenance is missing required fields: {missing}")
    if metadata.get("base_model") != expected_model_id or manifest.get("base_model") != expected_model_id:
        raise RuntimeError(f"Predictor base model must be {expected_model_id!r}.")
    if metadata.get("base_model_revision") != expected_revision or manifest.get("base_model_revision") != expected_revision:
        raise RuntimeError(f"Predictor base model revision must be {expected_revision!r}.")
    if metadata.get("problem_type") != "regression" or manifest.get("problem_type") != "regression":
        raise RuntimeError("Predictor artifact is not a regression artifact.")
    if metadata.get("weights_sha256") != WEIGHTS_SHA256 or metadata.get("config_sha256") != CONFIG_SHA256:
        raise RuntimeError("Predictor provenance does not identify the pinned Mitra Regressor bytes.")

    declared = manifest.get("files")
    if not isinstance(declared, list) or not declared:
        raise RuntimeError("Artifact manifest contains no file inventory.")
    entries: dict[str, dict[str, Any]] = {}
    for entry in declared:
        if not isinstance(entry, dict):
            raise RuntimeError("Artifact manifest file entries must be objects.")
        rel = str(entry.get("path", ""))
        member = PurePosixPath(rel)
        if not rel or "\\" in rel or member.is_absolute() or ".." in member.parts or rel in entries:
            raise RuntimeError(f"Unsafe or duplicate artifact manifest path: {rel!r}")
        entries[rel] = entry

    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path != root / ARTIFACT_MANIFEST
    }
    if actual != set(entries):
        missing_files = sorted(set(entries) - actual)
        unexpected_files = sorted(actual - set(entries))
        raise RuntimeError(
            f"Artifact inventory mismatch; missing={missing_files}, unexpected={unexpected_files}"
        )

    for rel, entry in entries.items():
        path = root / rel
        expected_size = int(entry.get("size", -1))
        expected_digest = _assert_hex_digest(str(entry.get("sha256", "")), f"sha256 for {rel}")
        if path.stat().st_size != expected_size:
            raise RuntimeError(f"Artifact file size mismatch: {rel}")
        if sha256_file(path) != expected_digest:
            raise RuntimeError(f"Artifact file digest mismatch: {rel}")

    if not (root / "predictor.pkl").is_file():
        raise RuntimeError("Artifact does not contain the required root-level AutoGluon predictor.pkl.")
    return manifest, metadata


# ---------------------------------------------------------------------------
# Fleet snapshot scheme (NOTEBOOK_SPEC 1.1 ST3/ST4, MOD13): manifest-driven verification and staging. The existing
# `stage_verified_hf_snapshot` (digest check + offline HF cache staging) stays the loader path and is called by
# `MitraRegressionPipeline.from_pretrained` after the manifest has been verified.
# ---------------------------------------------------------------------------


def verify_snapshot(path: str | Path | None = None) -> dict[str, Any]:
    """Check a local pinned snapshot against its manifest; raise naming the first mismatch.

    The manifest entries for ``model.safetensors`` and ``config.json`` must equal the package's own
    ``WEIGHTS_SHA256`` / ``CONFIG_SHA256`` constants, so the two can never diverge silently.
    """
    root = Path(path or DEFAULT_WEIGHTS_DIR)
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"snapshot manifest not found: {manifest_path}")
    with open(manifest_path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("modelId") != MODEL_ID:
        raise ValueError(f"manifest modelId {manifest.get('modelId')!r} != {MODEL_ID!r}")
    if manifest.get("revision") != MODEL_REVISION:
        raise ValueError(f"manifest revision {manifest.get('revision')!r} != {MODEL_REVISION!r}")
    entries = manifest.get("files", [])
    declared = {entry["path"]: entry["sha256"] for entry in entries}
    for filename, expected in ((WEIGHTS_FILE, WEIGHTS_SHA256), (CONFIG_FILE, CONFIG_SHA256)):
        if declared.get(filename) != expected:
            raise ValueError(
                f"manifest {filename} sha256 {declared.get(filename)!r} != package constant {expected!r}"
            )
    for entry in entries:
        file_path = root / entry["path"]
        if not file_path.is_file():
            raise FileNotFoundError(f"snapshot file missing: {file_path}")
        size = file_path.stat().st_size
        if size != entry["bytes"]:
            raise ValueError(f"{entry['path']}: size {size} != manifest {entry['bytes']}")
        digest = sha256_file(file_path)
        if digest != entry["sha256"]:
            raise ValueError(f"{entry['path']}: sha256 {digest} != manifest {entry['sha256']}")
    return {"path": str(root), **manifest}


def _hub_download(relative_path: str, root: Path) -> None:
    """Fetch one manifest-listed file at MODEL_REVISION straight into the snapshot directory."""
    from huggingface_hub import hf_hub_download

    hf_hub_download(repo_id=MODEL_ID, filename=relative_path, revision=MODEL_REVISION, local_dir=str(root))


def stage_missing_files(
    path: str | Path | None = None,
    *,
    allow_download: bool = False,
    downloader: Callable[[str, Path], None] | None = None,
) -> list[str]:
    """Fetch manifest-listed files that are absent locally (a clone commits the manifest but git-ignores the
    weights). Returns the relative paths fetched; ``verify_snapshot`` still runs after."""
    root = Path(path) if path is not None else DEFAULT_WEIGHTS_DIR
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    with open(manifest_path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("modelId") != MODEL_ID or manifest.get("revision") != MODEL_REVISION:
        raise ValueError(
            f"manifest names {manifest.get('modelId')}@{manifest.get('revision')}, "
            f"package pins {MODEL_ID}@{MODEL_REVISION}; refusing to stage"
        )
    missing = [entry["path"] for entry in manifest["files"] if not (root / entry["path"]).is_file()]
    if not missing:
        return []
    if not allow_download:
        raise FileNotFoundError(
            f"snapshot at {root} is missing {missing}; "
            f"pass allow_download=True to fetch them at {MODEL_REVISION}"
        )
    fetch = downloader or _hub_download
    for relative_path in missing:
        fetch(relative_path, root)
    return missing


class MitraRegressionPipeline:
    """Serving wrapper: the digest-verified pinned snapshot behind AutoGluon's Mitra regressor.

    ``from_pretrained`` stages and verifies the snapshot, then stages the verified bytes as the immutable offline
    Hugging Face snapshot AutoGluon resolves (``stage_verified_hf_snapshot``: HF_HUB_OFFLINE, no network path).
    ``fit`` registers the support rows through ``fit_mitra_predictor`` (in-context; gradient fine-tuning only with
    ``fine_tune=True``), ``evaluate`` and ``predict`` go through the repository's AutoGluon helpers.
    """

    def __init__(
        self,
        *,
        weights_path: Path,
        config_path: Path,
        snapshot_path: Path,
        device: str,
        source: str = "local-snapshot",
        predictor: Any = None,
        features: Sequence[str] | None = None,
    ) -> None:
        self.model_weight_path = Path(weights_path)
        self.config_path = Path(config_path)
        self.snapshot_path = Path(snapshot_path)
        self.device = device
        self.source = source
        self.predictor = predictor
        self.features: list[str] = list(features or [])
        self.target_column: str | None = None

    @classmethod
    def from_pretrained(
        cls,
        weights_dir: str | Path | None = None,
        *,
        allow_download: bool = False,
        hf_home: str | Path | None = None,
        device: str | None = None,
    ) -> MitraRegressionPipeline:
        root = Path(weights_dir or DEFAULT_WEIGHTS_DIR)
        stage_missing_files(root, allow_download=allow_download)
        verify_snapshot(root)
        home = Path(hf_home) if hf_home is not None else root / ".cache" / "hf"
        snapshot = stage_verified_hf_snapshot(root / WEIGHTS_FILE, root / CONFIG_FILE, hf_home=home)
        if device is None:
            try:
                import torch

                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"
        return cls(
            weights_path=root / WEIGHTS_FILE, config_path=root / CONFIG_FILE, snapshot_path=snapshot, device=device
        )

    def fit(
        self,
        train_data: pd.DataFrame,
        *,
        target_column: str,
        eval_metric: str,
        path: str | Path,
        fine_tune: bool = False,
        time_limit: int = 300,
        seed: int = 42,
        fine_tune_steps: int | None = None,
        max_memory_usage_ratio: float = 1.10,
    ) -> MitraRegressionPipeline:
        self.predictor = fit_mitra_predictor(
            train_data,
            target_column=target_column,
            eval_metric=eval_metric,
            path=path,
            fine_tune=fine_tune,
            time_limit=time_limit,
            seed=seed,
            fine_tune_steps=fine_tune_steps,
            max_memory_usage_ratio=max_memory_usage_ratio,
        )
        self.target_column = target_column
        self.features = [column for column in train_data.columns if column != target_column]
        self.source = "fine-tuned" if fine_tune else self.source
        return self

    def evaluate(self, frame: pd.DataFrame) -> dict[str, float]:
        """AutoGluon's evaluation of a labelled frame, normalised to positive error values."""
        if self.predictor is None:
            raise RuntimeError("Pipeline is not fitted; call fit(...) first")
        raw = self.predictor.evaluate(frame, auxiliary_metrics=True, silent=True)
        return normalize_autogluon_regression_metrics(raw)

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        if self.predictor is None:
            raise RuntimeError("Pipeline is not fitted; call fit(...) first")
        return predict_regression(self.predictor, frame, self.features)


# ---------------------------------------------------------------------------
# Role stages (DAT24 / EVAL21) on top of the existing validation and metric helpers.
# ---------------------------------------------------------------------------

INPUT_SCHEMA: dict[str, Any] = {
    "input": "pandas.DataFrame, one row per example; feature columns of any dtype plus a numeric target",
    "columns": "unique names; `drop_columns` are removed before validation",
    "target": (
        "coerced to numeric (non-numeric values are rejected); rows with a missing target are dropped and "
        "counted; infinite values are rejected; the training target must vary"
    ),
    "train_rows": [MIN_TRAIN_ROWS, MAX_TRAIN_ROWS],
    "eval_rows": [2, None],
    "features": [1, MAX_FEATURES],
    "inference_input": "every fitted feature column present; no `prediction` column; extras pass through",
    "preprocessing": (
        "none by the package (AutoGluon's Mitra handles raw columns); training rows above MAX_TRAIN_ROWS are "
        "capped by seeded sampling and the cap is reported"
    ),
}


def training_mean_baseline(
    train_targets: Iterable[float], holdout_targets: Iterable[float]
) -> dict[str, float]:
    """The trivial baseline: always predict the training mean (mae, rmse, r2: the `regression_metrics` ids)."""
    train = np.asarray(list(train_targets), dtype=float)
    holdout = np.asarray(list(holdout_targets), dtype=float)
    if train.size == 0 or holdout.size == 0 or not (np.isfinite(train).all() and np.isfinite(holdout).all()):
        raise ValueError("targets must be non-empty and finite")
    return regression_metrics(holdout, np.full(holdout.shape, float(train.mean())))


def validate_inputs(
    frame: pd.DataFrame,
    target_column: str | None = "target",
    *,
    drop_columns: Iterable[str] = (),
    min_rows: int = MIN_TRAIN_ROWS,
    require_variation: bool = True,
    feature_columns: Iterable[str] | None = None,
    names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Validation stage: return the input manifest (schema, observed table properties, verdict).

    With a ``target_column`` the table is checked exactly as ``validate_labeled_frame`` checks it (its report:
    dropped missing-target rows, exact duplicates, feature count — is carried, not hidden); with
    ``target_column=None`` it is an inference table checked exactly as ``validate_inference_frame`` checks it.
    Rejection is reported by raising the same error the core helper raises.
    """
    if names is not None and len(names) != 1:
        raise ValueError("names must have exactly one entry (the table's id)")
    table_id = names[0] if names else "table-0"
    if target_column is None:
        if feature_columns is None:
            raise ValueError("feature_columns is required to validate an inference table")
        checked, extra = validate_inference_frame(frame, feature_columns)
        entry: dict[str, Any] = {
            "id": table_id,
            "mode": "inference",
            "rows": len(checked),
            "feature_columns": list(checked.columns),
            "extra_columns": extra,
            "missing_value_columns": {str(c): int(n) for c, n in checked.isna().sum().items() if n > 0},
        }
    else:
        clean, features, report = validate_labeled_frame(
            frame,
            target_column,
            name=table_id,
            drop_columns=drop_columns,
            min_rows=min_rows,
            require_variation=require_variation,
        )
        values = clean[target_column].to_numpy(dtype=float)
        entry = {
            "id": table_id,
            "mode": "fit",
            "rows": len(clean),
            "feature_columns": features,
            "categorical_columns": [c for c in features if not pd.api.types.is_numeric_dtype(clean[c])],
            "missing_value_columns": {
                str(c): int(n) for c, n in clean[features].isna().sum().items() if n > 0
            },
            "report": report,
            "target_summary": {
                "min": float(values.min()),
                "max": float(values.max()),
                "mean": float(values.mean()),
                "distinct": int(clean[target_column].nunique()),
            },
        }
    return {
        "schema": dict(INPUT_SCHEMA),
        "inputs": [entry],
        "target_column": target_column,
        "drop_columns": list(drop_columns),
        "min_rows": min_rows if target_column is not None else None,
        "verdict": "accepted",
        "findings": [],
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
    }


def evaluation_report(
    metrics: Mapping[str, float] | None,
    *,
    baseline: Mapping[str, float] | None = None,
    independent_test: Mapping[str, float] | None = None,
    n_holdout: int | None = None,
    n_test: int | None = None,
    target_column: str | None = None,
    selection: str | None = None,
    sample_kind: str = "sample",
    estimation: str = "single seeded split; no dispersion estimate",
) -> dict[str, Any]:
    """Evaluation stage: a machine-readable report even when nothing is measurable.

    ``metrics`` / ``independent_test`` are dicts from ``regression_metrics`` (mae, rmse, r2) and ``baseline``
    from ``training_mean_baseline``; the verdict is ``sample-sanity``. Without metrics (no labelled rows) the
    verdict is ``not-measurable`` and the report says what labelled data would make the task measurable.
    """
    units = {"mae": "target units", "rmse": "target units", "r2": "unitless"}

    def _entries(source: Mapping[str, float]) -> list[dict[str, Any]]:
        unknown = sorted(set(source) - set(METRIC_IDS))
        if unknown:
            raise ValueError(f"unknown metric ids {unknown}; regression_metrics reports {list(METRIC_IDS)}")
        return [
            {
                "id": metric_id,
                "value": None if not np.isfinite(float(source[metric_id])) else float(source[metric_id]),
                "units": units[metric_id],
                "higher_is_better": metric_id == "r2",
            }
            for metric_id in METRIC_IDS
            if metric_id in source
        ]

    base: dict[str, Any] = {
        "task": "tabular regression by in-context conditioning on labelled support rows (AutoGluon Mitra)",
        "score_semantics": "continuous point predictions in target units; no per-prediction uncertainty",
        "sample_kind": sample_kind,
        "n_holdout": n_holdout,
        "n_test": n_test,
        "target_column": target_column,
        "selection": selection,
        "baselines": [],
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
    }
    if metrics is None:
        return {
            **base,
            "metrics": [],
            "independent_test": [],
            "verdict": "not-measurable",
            "reason": "no labelled holdout rows were supplied for the scored table",
            "needs": (
                "a labelled holdout table with a finite, varying numeric target column, scored with "
                "`regression_metrics` (mae, rmse, r2) against `training_mean_baseline`; an independent test "
                "partition from the deployment domain for any generalisable claim"
            ),
        }
    reported = [{**entry, "estimation": estimation} for entry in _entries(metrics)]
    test_entries: list[dict[str, Any]] = []
    if independent_test is not None:
        test_estimation = "independent test partition, single run"
        test_entries = [{**e, "estimation": test_estimation} for e in _entries(independent_test)]
    baselines = [] if baseline is None else [{"id": "training_mean", "metrics": _entries(baseline)}]
    rows = "an unstated number of" if n_holdout is None else str(n_holdout)
    return {
        **base,
        "metrics": reported,
        "independent_test": test_entries,
        "baselines": baselines,
        "verdict": "sample-sanity",
        "reason": f"{rows} labelled holdout row(s) from one seeded split; tutorial evidence, not a benchmark",
        "needs": (
            "an independent, domain-representative labelled test set for any generalisable quality claim; "
            "the point predictions carry no uncertainty interval"
        ),
    }
