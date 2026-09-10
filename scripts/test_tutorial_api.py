#!/usr/bin/env python3
"""Lightweight regression tests for the public tutorial API's validation/security boundary."""

from __future__ import annotations

import json
import sys
import tempfile
import warnings
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mitra_pipeline import (
    ARTIFACT_FORMAT,
    ARTIFACT_FORMAT_VERSION,
    CONFIG_SHA256,
    MODEL_ID,
    PINNED_REVISION,
    WEIGHTS_SHA256,
    read_csv_bytes,
    safe_extract_archive,
    validate_artifact_directory,
    validate_inference_frame,
    validate_labeled_frame,
    write_artifact_manifest,
)


def expect_raises(fn, contains: str) -> None:
    try:
        fn()
    except Exception as exc:  # noqa: BLE001
        if contains.lower() not in str(exc).lower():
            raise AssertionError(f"Expected {contains!r} in {exc!r}") from exc
    else:
        raise AssertionError(f"Expected failure containing {contains!r}")


def base_metadata() -> dict:
    return {
        "artifact_format": ARTIFACT_FORMAT,
        "artifact_format_version": ARTIFACT_FORMAT_VERSION,
        "base_model": MODEL_ID,
        "base_model_revision": PINNED_REVISION,
        "weights_sha256": WEIGHTS_SHA256,
        "config_sha256": CONFIG_SHA256,
        "autogluon_version": "1.5.0",
        "python_version": "3.12.0",
        "problem_type": "regression",
        "target_column": "target",
        "features": ["a", "b"],
        "mode": "pretrained",
        "selection_basis": "default:pretrained",
    }


def test_csv_and_regression_validation() -> None:
    duplicate = b"a,a,target\n1,2,3\n"
    expect_raises(lambda: read_csv_bytes(duplicate, "dup.csv"), "duplicate column")

    frame = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"], "target": [1.0, 2.0, 3.0]})
    clean, features, report = validate_labeled_frame(
        frame,
        "target",
        name="train.csv",
        min_rows=2,
        require_variation=True,
    )
    assert list(clean.columns) == ["a", "b", "target"]
    assert features == ["a", "b"]
    assert report["dropped_missing_target_rows"] == 0

    bad = frame.copy()
    bad.loc[0, "target"] = float("inf")
    expect_raises(
        lambda: validate_labeled_frame(bad, "target", name="bad.csv", min_rows=2),
        "infinite",
    )

    X, extra = validate_inference_frame(pd.DataFrame({"id": [7], "b": ["x"], "a": [1]}), ["a", "b"])
    assert list(X.columns) == ["a", "b"]
    assert extra == ["id"]


def test_artifact_manifest_roundtrip_and_tamper_detection() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "predictor"
        root.mkdir()
        (root / "predictor.pkl").write_bytes(b"trusted-fixture")
        (root / "model.bin").write_bytes(b"model-bytes")
        (root / "tutorial_run_metadata.json").write_text(json.dumps(base_metadata()), encoding="utf-8")

        write_artifact_manifest(root)
        manifest, metadata = validate_artifact_directory(root)
        assert manifest["artifact_format"] == ARTIFACT_FORMAT
        assert metadata["base_model_revision"] == PINNED_REVISION

        (root / "model.bin").write_bytes(b"tampered")
        expect_raises(lambda: validate_artifact_directory(root), "size mismatch")


def test_archive_path_and_expansion_guards() -> None:
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        bad = td / "bad.zip"
        with zipfile.ZipFile(bad, "w") as zf:
            zf.writestr("../escape.txt", "no")
        preserved = td / "out"
        preserved.mkdir()
        sentinel = preserved / "keep.txt"
        sentinel.write_text("keep", encoding="utf-8")
        expect_raises(lambda: safe_extract_archive(bad, preserved), "unsafe archive member")
        assert sentinel.read_text(encoding="utf-8") == "keep"

        duplicate = td / "duplicate.zip"
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=r"Duplicate name: .*", category=UserWarning)
            with zipfile.ZipFile(duplicate, "w") as zf:
                zf.writestr("same.txt", "first")
                zf.writestr("same.txt", "second")
        expect_raises(lambda: safe_extract_archive(duplicate, td / "out-duplicate"), "duplicate archive member")

        backslash = td / "backslash.zip"
        with zipfile.ZipFile(backslash, "w") as zf:
            zf.writestr("a\\b.txt", "no")
        expect_raises(lambda: safe_extract_archive(backslash, td / "out2"), "backslash")

        symlink = td / "symlink.zip"
        info = zipfile.ZipInfo("link")
        info.create_system = 3
        info.external_attr = (0o120777 << 16)
        with zipfile.ZipFile(symlink, "w") as zf:
            zf.writestr(info, "target")
        expect_raises(lambda: safe_extract_archive(symlink, td / "out3"), "symlink")


def main() -> int:
    test_csv_and_regression_validation()
    test_artifact_manifest_roundtrip_and_tamper_detection()
    test_archive_path_and_expansion_guards()
    print("Public tutorial API tests: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
