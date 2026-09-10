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
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

import numpy as np
import pandas as pd

MODEL_ID = "autogluon/mitra-regressor"
PINNED_REVISION = "5f277aa8f69042d39d6ac3612aed18bb9279bd95"
WEIGHTS_SHA256 = "d8e75c62af0bec2fd404b0ad20a442d951d43ca6d331315cfcc0509b54f2c642"
CONFIG_SHA256 = "2bc1ed5047f7c25368245e8ad32540a5fa28940b1ec05d3f1f454a09ff5384c1"

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
