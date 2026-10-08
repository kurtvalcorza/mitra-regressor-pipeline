"""Regression tests for the 2026-10-02 Notebook Review Framework v1 findings on
mitra_regressor_predictor_inference_colab (MRP-B1, MRP-M1..M2, MRP-m1..m4, MRP-m6;
docs/reviews/2026-10-02-notebook-review/mitra_regressor_predictor_inference_colab_Review.md).

CI's dependencies only (pandas, numpy, pytest): the notebook's own cells run with the carried module's pure-pandas
helpers and stand-ins for the download, extraction, AutoGluon and scikit-learn; no network, no torch. The one test that
compares the sample input with the real scikit-learn split is skipped where scikit-learn is absent.
"""
# ruff: noqa: E501

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
import re
import sys
import types
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "tutorials" / "mitra_regressor_predictor_inference_colab.ipynb"
WORKFLOW = ROOT / ".github" / "workflows" / "notebook-release.yml"
PINNED_URL = "https://github.com/kurtvalcorza/mitra-regressor-pipeline/releases/download/sample-bundle-v1/mitra_regressor_predictor.zip"
PINNED_SHA = "1b0892040f59a8182e63bc4545b74594d5be72f6b79d97e8f5f9bdbd07f29c48"
# DATA_DIGEST printed by the producing run (E2E notebook blob 256708ee, 2026-10-08 Colab T4) for the default sample.
PRODUCER_DATA_SHA256 = "0111b4ba11f24d2c8b118c163b2c32d907f957f8c5ea7142d1231a90ccfe5777"
FEATURES = ["age", "sex", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6"]


def _cells(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))["cells"]


def _code(path: Path, marker: str) -> str:
    found = [c["source"] for c in _cells(path) if c["cell_type"] == "code" and marker in c["source"]]
    assert len(found) == 1, (marker, len(found))
    return found[0]


def _markdown(path: Path) -> str:
    return "\n".join(c["source"] for c in _cells(path) if c["cell_type"] == "markdown")


def _set(source: str, name: str, value) -> str:
    pattern = re.compile(rf"^{name} = .*?(  # @param.*)?$", re.M)
    assert pattern.search(source), name
    return pattern.sub(lambda m: f"{name} = {value!r}" + (m.group(1) or ""), source, count=1)


def _api():
    spec = importlib.util.spec_from_file_location("_mrp_tutorial_api", ROOT / "mitra_pipeline" / "tutorial_api.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --- MRP-B1: the default path downloads the pinned sample bundle and verifies it before extraction --------------------


def test_mrp_b1_sample_artifact_is_pinned_by_release_url_sha256_and_producer():
    source = _code(ART, "SAMPLE_ARTIFACT = {")
    pinned = ast.literal_eval(re.search(r"^SAMPLE_ARTIFACT = (\{.*\})$", source, re.M).group(1))
    assert pinned["url"] == PINNED_URL and pinned["sha256"] == PINNED_SHA
    producer = pinned["producer"]
    assert producer["notebook"] == "tutorials/mitra_regressor_colab.ipynb" and producer["release"] == "sample-bundle-v1"
    assert re.fullmatch(r"[0-9a-f]{40}", producer["notebook_blob"]) and re.fullmatch(r"[0-9a-f]{40}", producer["commit"])
    assert "Colab" in producer["run"] and producer["evidence"].startswith("docs/execution-evidence/")
    assert producer["python"].startswith("3.12") and producer["autogluon"] == "1.5.0"  # a bundle is bound to these
    assert "ARTIFACT_ZIP_PATH = ''  # @param" in source  # the location field defaults to the pinned asset
    markdown = _markdown(ART)
    assert "Known NOTEBOOK_SPEC 2.0 gap" not in markdown and "no sample is bundled" not in markdown


def _run_section_4(tmp_path, monkeypatch, payload: bytes, *, pinned_sha: str | None = None, **fields):
    monkeypatch.chdir(tmp_path)
    source = _code(ART, "SAMPLE_ARTIFACT = {")
    if pinned_sha is not None:
        source = source.replace(PINNED_SHA, pinned_sha)
    for name, value in fields.items():
        source = _set(source, name, value)
    calls = {"download": [], "extract": []}

    def _download(url, destination, *, timeout=30):
        calls["download"].append(url)
        Path(destination).write_bytes(payload)

    def safe_extract_archive(zip_path, root):
        calls["extract"].append(str(zip_path))

    metadata = {"base_model": "m", "base_model_revision": "r", "weights_sha256": "w", "config_sha256": "c", "features": ["a"], "target_column": "target",
                "problem_type": "regression", "mode": "pretrained", "selection_basis": "default:pretrained", "autogluon_version": "1.5.0", "python_version": "3.12.12"}
    ns = {"os": os, "Path": Path, "sha256_file": lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest(), "_download": _download,
          "safe_extract_archive": safe_extract_archive, "validate_artifact_directory": lambda root: ({"artifact_format": "f", "artifact_format_version": 1}, metadata),
          "MODEL_ID": "m", "MODEL_REVISION": "r", "WEIGHTS_SHA256": "w", "CONFIG_SHA256": "c"}
    exec(compile(source, "section4", "exec"), ns)
    return ns, calls


def test_mrp_b1_default_path_downloads_verifies_then_extracts(tmp_path, monkeypatch):
    payload = b"PK sample bundle"
    ns, calls = _run_section_4(tmp_path, monkeypatch, payload, pinned_sha=hashlib.sha256(payload).hexdigest())
    assert calls["download"] == [PINNED_URL] and len(calls["extract"]) == 1
    assert ns["artifact_source"] == f"sample artifact: {PINNED_URL}" and ns["expected_digest"] == ns["zip_sha256"]
    ns, calls = _run_section_4(tmp_path, monkeypatch, payload, pinned_sha=hashlib.sha256(payload).hexdigest())
    assert calls["download"] == [] and len(calls["extract"]) == 1  # a complete earlier download is reused


def test_mrp_b1_a_substituted_sample_fails_before_extraction(tmp_path, monkeypatch):
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        _run_section_4(tmp_path, monkeypatch, b"tampered", pinned_sha=hashlib.sha256(b"original").hexdigest())
    assert not (tmp_path / "external-artifact" / "bundle").exists()


def test_mrp_m3_upload_without_colab_names_the_path_field(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "google", None)
    with pytest.raises(RuntimeError, match="needs the Colab upload dialog.*set ARTIFACT_ZIP_PATH to its path"):
        _run_section_4(tmp_path, monkeypatch, b"x", ARTIFACT_ZIP_PATH="upload")
    with pytest.raises(FileNotFoundError, match="ARTIFACT_ZIP_PATH 'missing.zip' is not a file"):
        _run_section_4(tmp_path, monkeypatch, b"x", ARTIFACT_ZIP_PATH="missing.zip")
    assert "Kaggle dataset, Google Drive or `gsutil` copy" in _markdown(ART)


# --- MRP-B1 sample input / MRP-M1 / MRP-m2 / MRP-m4: Section 6 ---------------------------------------------------------


def _fake_sklearn(monkeypatch, frame: pd.DataFrame):
    """Deterministic stand-ins for load_diabetes / train_test_split (CI has no scikit-learn)."""

    def load_diabetes(as_frame=True):
        return types.SimpleNamespace(frame=frame.copy(), target=types.SimpleNamespace(name="target"), data=frame.drop(columns=["target"]))

    def train_test_split(data, test_size, random_state=None, **_):
        cut = int(round(len(data) * (1 - test_size)))
        return data.iloc[:cut], data.iloc[cut:]

    sk = types.ModuleType("sklearn")
    datasets = types.ModuleType("sklearn.datasets")
    datasets.load_diabetes = load_diabetes
    selection = types.ModuleType("sklearn.model_selection")
    selection.train_test_split = train_test_split
    for name, module in {"sklearn": sk, "sklearn.datasets": datasets, "sklearn.model_selection": selection}.items():
        monkeypatch.setitem(sys.modules, name, module)


def _table(rows: int = 50) -> pd.DataFrame:
    frame = pd.DataFrame({c: [round(0.001 * (i + 1) * (k + 1), 6) for i in range(rows)] for k, c in enumerate(FEATURES)})
    frame["target"] = [float(100 + i) for i in range(rows)]
    return frame


def _digest(frame: pd.DataFrame) -> str:
    return hashlib.sha256(json.dumps({"sample.csv": hashlib.sha256(frame.to_csv(index=False, lineterminator="\n").encode("utf-8")).hexdigest()}, sort_keys=True).encode()).hexdigest()


def _run_section_6(monkeypatch, run_metadata: dict, numeric=FEATURES, **fields):
    api = _api()
    source = _code(ART, "NEW_DATA_PATH = ''  # @param")
    for name, value in fields.items():
        source = _set(source, name, value)
    predictor = types.SimpleNamespace(feature_metadata_in=types.SimpleNamespace(get_features=lambda valid_raw_types=None: list(numeric)))
    ns = {"os": os, "Path": Path, "json": json, "pd": pd, "TARGET_COLUMN": "target", "FEATURE_COLUMNS": FEATURES, "predictor": predictor,
          "run_metadata": run_metadata, "read_csv_bytes": api.read_csv_bytes, "validate_inputs": api.validate_inputs, "validate_inference_frame": api.validate_inference_frame}
    os.makedirs("outputs", exist_ok=True)
    exec(compile(source, "section6", "exec"), ns)
    return ns


def test_mrp_b1_sample_input_is_the_held_out_partition_and_limits_print_first(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    frame = _table()
    _fake_sklearn(monkeypatch, frame)
    ns = _run_section_6(monkeypatch, {"data_source": "Sample: Diabetes", "seed": 42, "data_sha256": _digest(frame), "support_rows": 30})
    out = capsys.readouterr().out
    assert ns["sample_kind"] == "sample" and len(ns["new_data"]) == 10 and "target" not in ns["new_data"].columns
    assert list(ns["new_data"]["age"]) == list(frame["age"].iloc[40:])  # the last split's held-out half, never the support rows
    assert out.index("'inference_limits'") < out.index("'sample_input'") and "'overlap_with_support_rows': 0" in out
    assert "MIN_TRAIN_ROWS" not in _code(ART, "NEW_DATA_PATH = ''  # @param")  # producer ceilings are no longer printed as limits


def test_mrp_b1_sample_input_refuses_a_different_table_or_bundle(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    frame = _table()
    _fake_sklearn(monkeypatch, frame)
    with pytest.raises(RuntimeError, match="fits only a bundle trained on 'Sample: Diabetes'"):
        _run_section_6(monkeypatch, {"data_source": "Upload CSV", "seed": 42})
    with pytest.raises(RuntimeError, match="is not the table the bundle was produced from"):
        _run_section_6(monkeypatch, {"data_source": "Sample: Diabetes", "seed": 42, "data_sha256": "0" * 64})
    with pytest.raises(RuntimeError, match="does not match the bundle"):
        _run_section_6(monkeypatch, {"data_source": "Sample: Diabetes", "seed": 42, "data_sha256": _digest(frame), "support_rows": 31})


def test_mrp_b1_sample_input_matches_the_real_e2e_test_partition(tmp_path, monkeypatch):
    sklearn_datasets = pytest.importorskip("sklearn.datasets")
    from sklearn.model_selection import train_test_split

    monkeypatch.chdir(tmp_path)
    frame = sklearn_datasets.load_diabetes(as_frame=True).frame
    metadata = {"data_source": "Sample: Diabetes", "seed": 42, "data_sha256": PRODUCER_DATA_SHA256, "support_rows": 265}
    if _digest(frame) != PRODUCER_DATA_SHA256:
        # Another scikit-learn/pandas than the lock's (1.7.2 / 2.3.3) serialises the table differently: the notebook
        # must refuse rather than score rows it cannot tie to the producer's data digest.
        with pytest.raises(RuntimeError, match="is not the table the bundle was produced from"):
            _run_section_6(monkeypatch, metadata)
        return
    ns = _run_section_6(monkeypatch, {"data_source": "Sample: Diabetes", "seed": 42, "data_sha256": PRODUCER_DATA_SHA256, "support_rows": 265})
    support, remainder = train_test_split(frame, test_size=0.4, random_state=42)  # the E2E notebook's Section 4 split
    _, test = train_test_split(remainder, test_size=0.5, random_state=42)
    assert len(ns["new_data"]) == 89 and not set(test.index) & set(support.index)
    pd.testing.assert_frame_equal(ns["new_data"].reset_index(drop=True), test.drop(columns=["target"]).reset_index(drop=True), check_dtype=False, check_exact=False, rtol=1e-12)


def test_mrp_m1_rows_by_upload_work_when_the_bundle_came_by_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    rows = _table(5).drop(columns=["target"]).to_csv(index=False).encode("utf-8")
    colab = types.ModuleType("google.colab")
    colab.files = types.SimpleNamespace(upload=lambda: {"mine.csv": rows})
    google = types.ModuleType("google")
    google.colab = colab
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    _fake_sklearn(monkeypatch, _table())
    ns = _run_section_6(monkeypatch, {"data_source": "Sample: Diabetes", "seed": 42}, NEW_DATA_PATH="upload")  # NameError: files on the reviewed blob
    assert ns["csv_name"] == "mine.csv" and len(ns["new_data"]) == 5 and ns["sample_kind"] == "BYOD"


def test_mrp_m4_non_numeric_feature_is_refused_in_section_6_naming_the_column(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    bad = _table(5).drop(columns=["target"]).astype(object)
    bad.loc[2, "age"] = "abc"
    bad.to_csv(tmp_path / "bad.csv", index=False)
    _fake_sklearn(monkeypatch, _table())
    with pytest.raises(ValueError, match=r"feature 'age' must be numeric.*\['abc'\]"):
        _run_section_6(monkeypatch, {"data_source": "Sample: Diabetes", "seed": 42}, NEW_DATA_PATH=str(tmp_path / "bad.csv"))
    assert not (tmp_path / "outputs" / "mitra_regressor_predictor_inference_input_manifest.json").exists()  # not written as accepted
    source = _code(ART, "NEW_DATA_PATH = ''  # @param")
    assert source.index("must be numeric") < source.index("input_manifest = validate_inputs(")
    assert '"feature … must be numeric"' in _markdown(ART)


# --- MRP-m1 / MRP-M2 ----------------------------------------------------------------------------------------------------


def test_mrp_m1_prose_states_what_section_3_adds_and_that_section_5_reads_pipe():
    markdown = _markdown(ART)
    assert "Prediction needs only the bundle" in markdown and "proves that the pin the bundle names still resolves" in markdown
    assert "The pinned snapshot `pipe` of Section 3 is not used for inference." not in markdown
    assert "this cell needs Section 3 to have run" in markdown


def test_mrp_m2_in_notebook_activity_runs_on_the_default_sample_with_one_field():
    source = _code(ART, "ACTIVITY_SHIFT = ")
    assert "ACTIVITY_FEATURE = 'bmi'  # @param" in source and "ACTIVITY_SHIFT = 0.05  # @param" in source
    assert ".fit(" not in source and "to_csv(" not in source  # changes nothing above it
    markdown = _markdown(ART)
    assert "## 8. Activity: shift one feature" in markdown and "*Change* `ACTIVITY_SHIFT` to `-0.05` (one field)" in markdown
    assert markdown.count("<details><summary>Check your reasoning</summary>") == 4
    for element in ("**Who this notebook is for.**", "**How to use this notebook.**", "**Input → Model → Output.**", "## Troubleshooting", "## Glossary", "## Conclusion (your notes)"):
        assert element in markdown, element
    ns = {"FEATURE_COLUMNS": FEATURES, "X_new": _table(4)[FEATURES], "out": pd.DataFrame({"prediction": [100.0, 110.0, 120.0, 130.0]}), "pd": pd,
          "np": __import__("numpy"), "serving": types.SimpleNamespace(predict=lambda frame: 100.0 + 1000.0 * frame["bmi"].to_numpy() + [0.0, 10.0, 20.0, 30.0] - 1000.0 * _table(4)["bmi"].to_numpy())}
    exec(source, ns)


# --- MRP-m6: CI scores only rows outside the bundle's support set ------------------------------------------------------


def test_mrp_m6_ci_inference_rows_are_the_test_partition_with_an_overlap_assertion():
    workflow = WORKFLOW.read_text(encoding="utf-8")
    assert "data.tail(24)" not in workflow
    assert "_, test_rows = train_test_split(remainder, test_size=0.5, random_state=seed)" in workflow
    assert "raise RuntimeError(f\"inference rows overlap the bundle's support rows" in workflow


# --- ST1 exemption (fleet pattern, language-model-pipeline #78) ---------------------------------------------------------


def test_st1_allows_only_the_pinned_sample_bundle_asset_url():
    spec = importlib.util.spec_from_file_location("_validator", ROOT / "tools" / "validate_release_assets.py")
    validator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(validator)
    st1 = dict(validator.FORBIDDEN_PATTERNS)["repository clone (ST1)"]
    assert not st1.search(f"'{PINNED_URL}'")
    for bad in ("'https://github.com/kurtvalcorza/mitra-regressor-pipeline.git'", "'https://github.com/kurtvalcorza/mitra-regressor-pipeline/archive/main.zip'",
                "'https://github.com/kurtvalcorza/mitra-regressor-pipeline/releases/download/sample-bundle-v1/x.py'", "'https://github.com/kurtvalcorza/other-repo/releases/download/sample-bundle-v1/x.zip'"):
        assert st1.search(bad), bad
