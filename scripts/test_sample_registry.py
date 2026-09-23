"""Load every committed sample-dataset archive through the E2E notebook's data cell without network.

Regression guard for the Kaggle T4 run of 2026-09-07, where the data-loading cell
raised ``KeyError: 'urls'`` for every ``DATA_SOURCE``. Since the standalone
NOTEBOOK_SPEC 2.0 notebook, the default sample is scikit-learn's bundled diabetes
table and the committed ``examples/sample-data/*.zip`` archives are supplied through
the ``Upload pre-split train/val/test`` BYOD branch. The notebook's data-loading cell
is executed once per committed archive with ``google.colab.files.upload`` replaced by
a stub that serves the archive's ``train.csv``/``val.csv``/``test.csv`` and
``urllib.request.urlopen`` replaced by a stub that fails, so the test is hermetic and
also proves that each archive matches the target and row counts in its dataset card.
"""

from __future__ import annotations

import ast
import contextlib
import hashlib
import io
import json
import os
import re
import sys
import tempfile
import types
import unittest
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

NOTEBOOK = Path(os.environ.get("SAMPLE_REGISTRY_NOTEBOOK", ROOT / "tutorials" / "mitra_regressor_colab.ipynb"))
SAMPLE_DIR = ROOT / "examples" / "sample-data"
DATASET_CARD = SAMPLE_DIR / "DATASET_CARD.md"
PRESPLIT = "Upload pre-split train/val/test"
SPLIT_FILES = ("train.csv", "val.csv", "test.csv")

try:  # the static CI job installs only pandas + numpy; the cell imports sklearn for the sample and single-CSV branches only
    import sklearn.datasets  # noqa: F401
    import sklearn.model_selection  # noqa: F401
except ImportError:  # pragma: no cover - exercised only on minimal runners
    def _unavailable(*args, **kwargs):
        raise RuntimeError("scikit-learn is not installed on this runner")

    sys.modules.setdefault("sklearn", types.ModuleType("sklearn"))
    for name, attr in (("sklearn.datasets", "load_diabetes"), ("sklearn.model_selection", "train_test_split")):
        stub = types.ModuleType(name)
        setattr(stub, attr, _unavailable)
        sys.modules[name] = stub

import mitra_pipeline as mp  # noqa: E402


def load_data_cell() -> str:
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    for cell in payload["cells"]:
        source = cell["source"]
        text = "".join(source) if isinstance(source, list) else source
        if cell["cell_type"] == "code" and re.search(r"^DATA_SOURCE = ", text, re.MULTILINE):
            return "\n".join(line for line in text.splitlines() if not line.lstrip().startswith(("%", "!")))
    raise AssertionError(f"{NOTEBOOK.name}: no code cell defines the DATA_SOURCE form field")


def form_value(cell: str, name: str):
    for node in ast.parse(cell).body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError(f"{name} is not a top-level literal assignment")


def with_form(cell: str, **values) -> str:
    for name, value in values.items():
        pattern = re.compile(rf"^{name} = .*?(\s*# @param.*)?$", re.MULTILINE)
        assert pattern.search(cell), f"{name} form field not found"
        cell = pattern.sub(lambda m, n=name, v=value: f"{n} = {v!r}{m.group(1) or ''}", cell, count=1)
    return cell


def card_entries() -> dict[str, dict]:
    """Archive name -> target column and per-split row counts, as documented in DATASET_CARD.md."""
    entries: dict[str, dict] = {}
    for section in re.split(r"^## ", DATASET_CARD.read_text(encoding="utf-8"), flags=re.MULTILINE):
        archive = re.search(r"\*\*Archive:\*\* `([^`]+\.zip)`", section)
        if not archive:
            continue
        target = re.search(r"\*\*Target:\*\* `([^`]+)`", section)
        rows = re.search(r"\*\*Rows:\*\* ([\d,]+) train · ([\d,]+) val · ([\d,]+) test", section)
        assert target and rows, f"{archive.group(1)}: dataset card lacks a Target or Rows line"
        counts = [int(value.replace(",", "")) for value in rows.groups()]
        entries[archive.group(1)] = {"target": target.group(1), "rows": dict(zip(SPLIT_FILES, counts, strict=True))}
    return entries


def archive_members(path: Path) -> dict[str, bytes]:
    with zipfile.ZipFile(path) as archive:
        return {Path(name).name: archive.read(name) for name in archive.namelist() if Path(name).name in SPLIT_FILES}


def _offline(url, *args, **kwargs):
    raise urllib.error.URLError(f"network access attempted from the data cell: {url}")


def run_cell(cell: str, uploads: dict[str, bytes] | None = None, **form) -> dict:
    colab = types.ModuleType("google.colab")
    colab.files = types.SimpleNamespace(upload=lambda: dict(uploads or {}))
    google = types.ModuleType("google")
    google.colab = colab
    namespace = {
        "__name__": "__notebook__",
        "json": json,
        "os": os,
        "Path": Path,
        "read_csv_bytes": mp.read_csv_bytes,
        "display": lambda *args, **kwargs: None,
    }
    with tempfile.TemporaryDirectory() as scratch, mock.patch("urllib.request.urlopen", _offline), \
            mock.patch.dict(sys.modules, {"google": google, "google.colab": colab}):
        cwd = os.getcwd()
        os.chdir(scratch)  # Colab/Kaggle do not run from the repository root
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                exec(compile(with_form(cell, **form), f"{NOTEBOOK.name}:data-cell", "exec"), namespace)
        finally:
            os.chdir(cwd)
    return namespace


class SampleDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cell = load_data_cell()
        cls.card = card_entries()

    def test_default_path_is_the_bundled_sample(self) -> None:
        self.assertTrue(str(form_value(self.cell, "DATA_SOURCE")).startswith("Sample"))
        self.assertIs(form_value(self.cell, "USE_BYOD"), False)

    def test_dataset_card_documents_every_committed_archive(self) -> None:
        committed = {path.name for path in SAMPLE_DIR.glob("*.zip")}
        self.assertTrue(committed, "no committed sample archives")
        self.assertEqual(set(self.card), committed, "dataset card entries and committed archives differ")

    def test_every_archive_loads_through_presplit_upload_without_network(self) -> None:
        for archive, entry in self.card.items():
            with self.subTest(archive=archive):
                members = archive_members(SAMPLE_DIR / archive)
                self.assertEqual(set(members), set(SPLIT_FILES), archive)
                namespace = run_cell(self.cell, uploads=members, DATA_SOURCE=PRESPLIT, USE_BYOD=True, TARGET_COLUMN=entry["target"])
                for name, split in zip(("train_data", "holdout_data", "test_data"), SPLIT_FILES, strict=True):
                    frame = namespace[name]
                    self.assertEqual(len(frame), entry["rows"][split], f"{archive}:{split}")
                    self.assertIn(entry["target"], frame.columns, f"{archive}:{split}")
                expected = hashlib.sha256(json.dumps({name: hashlib.sha256(members[name]).hexdigest() for name in sorted(SPLIT_FILES)}, sort_keys=True).encode()).hexdigest()
                self.assertEqual(namespace["DATA_DIGEST"], expected)

    def test_incomplete_presplit_upload_is_rejected(self) -> None:
        members = archive_members(SAMPLE_DIR / next(iter(self.card)))
        members.pop("test.csv")
        with self.assertRaisesRegex(RuntimeError, "Missing"):
            run_cell(self.cell, uploads=members, DATA_SOURCE=PRESPLIT, USE_BYOD=True)

    def test_upload_selector_requires_byod_opt_in(self) -> None:
        with self.assertRaisesRegex(ValueError, "USE_BYOD"):
            run_cell(self.cell, DATA_SOURCE=PRESPLIT, USE_BYOD=False)


if __name__ == "__main__":
    unittest.main()
