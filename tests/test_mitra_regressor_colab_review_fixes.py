"""Regression tests for the 2026-10-02 Notebook Review Framework v1 findings on mitra_regressor_colab (MRC-M1..M2,
MRC-m1..m6; docs/reviews/2026-10-02-notebook-review/mitra_regressor_colab_Review.md).

CI's dependencies only: the notebook's own cell sources are executed with stand-ins; no AutoGluon, torch or network.
"""
# ruff: noqa: E501

from __future__ import annotations

import ast
import contextlib
import json
import logging
import os
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
E2E = ROOT / "tutorials" / "mitra_regressor_colab.ipynb"
ART = ROOT / "tutorials" / "mitra_regressor_predictor_inference_colab.ipynb"


def _cells(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))["cells"]


def _code(path: Path, marker: str) -> str:
    found = [c["source"] for c in _cells(path) if c["cell_type"] == "code" and marker in c["source"]]
    assert len(found) == 1, (marker, len(found))
    return found[0]


def _markdown(path: Path) -> str:
    return "\n".join(c["source"] for c in _cells(path) if c["cell_type"] == "markdown")


def _section6() -> str:
    return _code(E2E, "EVAL_METRIC = ")


# --- MRC-M1: Mitra's in-context support is printed, recorded and explained ---------------------------------------------


def test_mrc_m1_context_rows_are_printed_recorded_and_the_prose_is_truthful():
    source = _section6()
    markdown = _markdown(E2E)
    assert "exact same support rows" not in markdown and "(EVAL15)" not in markdown
    assert "`Validation score` line of its log is measured on those internal rows, **not** on your holdout" in markdown
    assert "`-48.1` means an MAE of 48.1 on those rows" in markdown and "212 of 265" in markdown
    assert "'mitra_context_rows': MITRA_CONTEXT_ROWS" in source and "'autogluon_internal_validation_rows'" in source
    export = _code(E2E, "archive_base = ")
    assert "'mitra_context_rows': context_rows(ACTIVE_MODEL)" in export and "'support_rows': len(train_data)" in export
    helper = source[source.index("def context_rows(") : source.index("with autogluon_log():\n    pipe.fit(")]
    ns: dict = {}
    exec(helper, ns)

    class _Predictor:
        def load_data_internal(self, data="train", return_X=True, return_y=True):
            assert data == "train" and return_y is False
            return list(range(212)), None

    assert ns["context_rows"](types.SimpleNamespace(predictor=_Predictor())) == 212
    assert ns["context_rows"](types.SimpleNamespace(predictor=None)) is None  # reported, never hidden


# --- MRC-m5: shortened AutoGluon log ------------------------------------------------------------------------------------


def test_mrc_m5_autogluon_log_summary_keeps_only_split_score_and_warning_lines(monkeypatch):
    source = _section6()
    assert "AUTOGLUON_LOG = 'summary'  # @param" in source
    start, end = source.index("class _AutoGluonSummary("), source.index("def context_rows(")
    monkeypatch.setitem(sys.modules, "autogluon", types.ModuleType("autogluon"))
    monkeypatch.setitem(sys.modules, "autogluon.tabular", types.ModuleType("autogluon.tabular"))
    ns = {"logging": logging, "contextlib": contextlib, "AUTOGLUON_LOG": "summary"}
    exec(source[start:end], ns)
    logger = logging.getLogger("autogluon")
    records: list[str] = []

    class _Collect(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    handler = _Collect()
    logger.addHandler(handler)
    child = logging.getLogger("autogluon.tabular.learner")
    old_level = child.level
    try:
        with ns["autogluon_log"]():
            child.setLevel(logging.INFO)
            child.info("Presets specified: ['extreme']  Massively better than 'best'")
            child.info("Automatically generating train/validation split with holdout_frac=0.2, Train Rows: 212, Val Rows: 53")
            child.info("This metric's sign has been flipped to adhere to being higher_is_better.")
            child.info("\t-48.1091\t = Validation score   (-mean_absolute_error)")
            child.warning("Not enough memory to train Mitra")
        assert records == [
            "Automatically generating train/validation split with holdout_frac=0.2, Train Rows: 212, Val Rows: 53",
            "\t-48.1091\t = Validation score   (-mean_absolute_error)",
            "Not enough memory to train Mitra",
        ]
        assert not handler.filters  # removed after the fit
    finally:
        logger.removeHandler(handler)
        child.setLevel(old_level)


# --- MRC-m2: the comparison is read across partitions and metrics ------------------------------------------------------


def test_mrc_m2_gap_is_printed_for_holdout_and_test_and_what_to_notice_follows():
    source = _section6()
    block = source[source.index("def mae_gap(") : source.index("print({'holdout_mae_gap_best_tree_minus_mitra'")]
    rows = [
        {"model": "LightGBM", "split": "holdout", "mae": 44.9}, {"model": "RandomForest", "split": "holdout", "mae": 42.3},
        {"model": "Mitra-pretrained", "split": "holdout", "mae": 40.1},
        {"model": "LightGBM", "split": "test", "mae": 46.0}, {"model": "RandomForest", "split": "test", "mae": 44.7},
        {"model": "Mitra-pretrained", "split": "test", "mae": 44.5},
    ]
    ns = {"baseline_rows": rows, "ACTIVE_MODE": "pretrained"}
    exec(block, ns)
    assert ns["mae_gap"]("holdout") == 2.2 and ns["mae_gap"]("test") == 0.2
    assert ns["mae_gap"]("missing") is None
    markdown = _markdown(E2E)
    notice = markdown[markdown.index("**What to notice.** (1) Compare `holdout_mae_gap_best_tree_minus_mitra`") :]
    assert "`test_mae_gap_best_tree_minus_mitra`" in notice and "Read RMSE and R² beside MAE" in notice
    assert "no dispersion estimate" in notice
    assert "The executable baselines show when the foundation model adds" not in markdown


# --- MRC-m4: EVAL_METRIC scope, both Mitra rows, GPU gate before any fit, next experiments with rerun ranges ----------


def test_mrc_m4_both_mitra_variants_reach_the_table_and_the_gpu_gate_runs_first():
    source = _section6()
    gate = source.index("if RUN_FINE_TUNING and not torch.cuda.is_available():")
    assert gate < source.index("pipe.fit(") and gate < source.index("candidate.fit(")
    block = source[source.index("mitra_variants = ") : source.index("metrics_table = pd.DataFrame(baseline_rows)")]
    pre, pre_t, ft, ft_t = {"mae": 40.0}, {"mae": 44.0}, {"mae": 39.0}, {"mae": 45.0}
    ns = {"pretrained_metrics": pre, "pretrained_test_metrics": pre_t, "candidate_metrics": ft, "candidate_test_metrics": ft_t,
          "candidate": object(), "baseline_rows": []}
    exec(block, ns)
    assert [(r["model"], r["split"]) for r in ns["baseline_rows"]] == [
        ("Mitra-pretrained", "holdout"), ("Mitra-pretrained", "test"), ("Mitra-fine-tuned", "holdout"), ("Mitra-fine-tuned", "test")]
    ns.update(candidate=None, baseline_rows=[])
    exec(block, ns)
    assert [r["model"] for r in ns["baseline_rows"]] == ["Mitra-pretrained", "Mitra-pretrained"]
    assert "# EVAL_METRIC sets AutoGluon's internal validation score and the fine-tune selection only" in source
    markdown = _markdown(E2E)
    assert "changing `EVAL_METRIC` does **not** change the metrics table" in markdown
    closing = markdown[markdown.index("## Change one thing (next experiments)") :]
    for text in ("*Run* — Sections 4–7", "re-run Sections 6–9", "re-run Sections 4–9", "*Predict*", "*Explain*", "**Activity (about 2 minutes on CPU)"):
        assert text in closing, text


# --- MRC-m3: limits before the upload, archives refused, TARGET_COLUMN named in Section 4 -----------------------------


def _section4_helpers():
    source = _code(E2E, "def byod_payloads(")
    helper = source[source.index("def byod_payloads(") : source.index("test_data = None")]
    ns = {"os": os, "Path": Path, "TARGET_COLUMN": "target"}
    exec(helper, ns)
    return source, ns


def test_mrc_m3_archives_are_refused_with_an_extract_instruction(tmp_path, monkeypatch):
    _, ns = _section4_helpers()
    archive = tmp_path / "insurance-medical-charges.zip"
    archive.write_bytes(b"PK")
    with pytest.raises(ValueError, match="is an archive: extract it"):
        ns["byod_payloads"](str(archive))
    with pytest.raises(ValueError, match="is an archive: extract it"):  # the pre-split option given the ZIP itself
        ns["byod_payloads"](str(archive), ("train.csv", "val.csv", "test.csv"))
    colab = types.ModuleType("google.colab")
    colab.files = types.SimpleNamespace(upload=lambda: {"insurance-medical-charges.zip": b"PK"})
    google = types.ModuleType("google")
    google.colab = colab
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    with pytest.raises(RuntimeError, match="Extract the archive and upload its CSV"):
        ns["byod_payloads"]("", ("train.csv", "val.csv", "test.csv"))


def test_mrc_m3_wrong_target_column_is_named_in_section_4_and_the_split_is_checked_before_upload():
    import pandas as pd

    source, ns = _section4_helpers()
    frame = pd.DataFrame({"age": [19], "charges": [16884.9]})
    with pytest.raises(ValueError, match=r"TARGET_COLUMN 'target' is not a column of train.csv. Set TARGET_COLUMN in this cell \(Section 4\).*'charges'"):
        ns["require_target"](frame, "train.csv")
    ns["TARGET_COLUMN"] = "charges"
    assert ns["require_target"](frame, "train.csv") is frame
    assert source.index("if DATA_SOURCE == 'Upload CSV' and not 0.05 <= VALIDATION_SPLIT <= 0.40:") < source.index("def byod_payloads(")
    cells = _cells(E2E)
    idx = next(i for i, c in enumerate(cells) if c["cell_type"] == "code" and "def byod_payloads(" in c["source"])
    before = cells[idx - 1]["source"]
    for text in ("**numeric**", "**50**", "**500**", "**10,000**", "extract a ZIP first", "`charges` for Insurance Charges", "`SalePrice` for Ames Housing"):
        assert text in before, text


# --- MRC-M2 / MRC-m1: guided layer, short opening, no reviewer-facing text ----------------------------------------------


def test_mrc_m1_short_learner_opening_and_no_open_decision_text():
    cells = _cells(E2E)
    assert cells[0]["cell_type"] == "markdown" and len(cells[0]["source"]) <= 1800
    assert cells[1]["source"].startswith("<details><summary><b>About this notebook</b>") and "**Run all:**" in cells[1]["source"]
    markdown = _markdown(E2E)
    assert "open decision" not in markdown and "a reviewer reading" not in markdown


def test_mrc_m2_guided_elements_and_memory_troubleshooting_are_present():
    markdown = _markdown(E2E)
    for element in ("**Who this notebook is for.**", "**How to use this notebook.**", "**Input → Model → Output.**", "**Roadmap:**",
                    "## Troubleshooting", "## Glossary", "## Conclusion (your notes)", "**What to notice.**", "**Activity"):
        assert element in markdown, element
    assert markdown.count("**Predict:**") >= 5 and markdown.count("<details><summary>Check your reasoning</summary>") >= 5
    trouble = markdown[markdown.index("## Troubleshooting") :]
    assert "No models were trained successfully" in trouble and "`MAX_MEMORY_USAGE_RATIO`" in trouble
    assert "negative `Validation score`" in trouble
    carrier = [c for c in _cells(E2E) if c["cell_type"] == "code" and c["metadata"].get("dimer", {}).get("embedded_module")]
    assert carrier and all(c["metadata"].get("cellView") == "form" for c in carrier)


def test_rel13_quoted_counts_are_labelled_with_their_source_revision():
    markdown = _markdown(E2E)
    assert "previous notebook revision (blob `d32d6f6`)" in markdown
    assert "the recorded hosted run of the previous version needed one" not in markdown
    assert "the recorded run printed `PASS`" not in markdown


# --- MRC-m6: the four status documents agree ---------------------------------------------------------------------------


def test_mrc_m6_status_documents_name_spec_2_2_and_the_same_evidence_state():
    status = (ROOT / "STATUS.md").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    tutorials = (ROOT / "tutorials" / "README.md").read_text(encoding="utf-8")
    release = (ROOT / "docs" / "release-verification.md").read_text(encoding="utf-8")
    for text in (status, readme, tutorials, release):
        assert "Specification 2.2" in text
    assert "Specification 1.1" not in status and "Specification 1.1 and are" not in readme
    assert "No clean-runtime execution of the standalone notebooks has been recorded yet; clean GPU" not in release
    assert "SAMPLE_CONFIGS" not in tutorials and "512 training rows" not in tutorials
    assert "the standalone pair has not been executed yet" not in readme


# --- MRC-S1 / spec declaration -----------------------------------------------------------------------------------------


@pytest.mark.parametrize("path", [E2E, ART])
def test_generated_notebooks_declare_notebook_spec_2_2(path):
    notebook = json.loads(path.read_text(encoding="utf-8"))
    assert notebook["metadata"]["dimer"]["notebook_spec"] == "2.2"
    assert "DIMER Notebook Specification 2.2 — **standalone**" in _markdown(path)


def test_section_4_and_6_cells_parse():
    ast.parse(_section6())
    ast.parse(_code(E2E, "def byod_payloads("))
