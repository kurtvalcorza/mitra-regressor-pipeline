"""Load every sample-dataset registry variant of the E2E notebook without network.

Regression guard for the Kaggle T4 run of 2026-09-07, where the data-loading cell
raised ``KeyError: 'urls'`` for every ``DATA_SOURCE`` (including the default) because
the registry entries did not carry the keys the resolution code read.

The notebook's data-loading cell is executed once per ``SAMPLE_CONFIGS`` selector
with ``urllib.request.urlopen`` replaced by a stub that serves the committed
``examples/sample-data/*.zip`` bytes, so the test is hermetic and also proves that
the registry digests match the committed archives.
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
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

NOTEBOOK = Path(os.environ.get("SAMPLE_REGISTRY_NOTEBOOK", ROOT / "tutorials" / "mitra_regressor_colab.ipynb"))
SAMPLE_DIR = ROOT / "examples" / "sample-data"
FORBIDDEN_URL_FRAGMENTS = ("feat/add-sample-datasets",)

try:  # the static CI job installs only pandas + numpy; the cell imports sklearn for the BYOD branch only
    import sklearn.model_selection  # noqa: F401
except ImportError:  # pragma: no cover - exercised only on minimal runners
    def _unavailable(*args, **kwargs):
        raise RuntimeError("scikit-learn is not installed on this runner")

    stub = types.ModuleType("sklearn.model_selection")
    stub.train_test_split = _unavailable
    sys.modules.setdefault("sklearn", types.ModuleType("sklearn"))
    sys.modules["sklearn.model_selection"] = stub

try:
    import mitra_pipeline as mp
except ImportError:  # pre-Spec-1.0 trees have no public package; the cell must not need it
    mp = None


def load_data_cell() -> str:
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    for cell in payload["cells"]:
        source = cell["source"]
        text = "".join(source) if isinstance(source, list) else source
        if cell["cell_type"] == "code" and "SAMPLE_CONFIGS = {" in text:
            return "\n".join(line for line in text.splitlines() if not line.lstrip().startswith(("%", "!")))
    raise AssertionError(f"{NOTEBOOK.name}: no code cell defines SAMPLE_CONFIGS")


def registry(cell: str) -> dict[str, dict]:
    for node in ast.parse(cell).body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "SAMPLE_CONFIGS" for t in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError("SAMPLE_CONFIGS is not a top-level literal assignment")


def with_selector(cell: str, selector: str) -> str:
    pattern = re.compile(r"^DATA_SOURCE = .*?(\s*# @param.*)?$", re.MULTILINE)
    assert pattern.search(cell), "DATA_SOURCE form field not found"
    return pattern.sub(lambda m: f"DATA_SOURCE = {selector!r}{m.group(1) or ''}", cell, count=1)


class ArchiveServer:
    """urlopen stand-in that serves committed sample archives and records every URL."""

    def __init__(self, tamper: bool = False) -> None:
        self.urls: list[str] = []
        self.tamper = tamper

    def __call__(self, url, timeout=None, *args, **kwargs):
        url = url if isinstance(url, str) else url.full_url
        self.urls.append(url)
        target = SAMPLE_DIR / url.rsplit("/", 1)[-1]
        if not target.is_file():
            raise urllib.error.URLError(f"no committed sample archive for {url}")
        payload = target.read_bytes() + (b"\0" if self.tamper else b"")
        return io.BytesIO(payload)


def run_cell(cell: str, selector: str, tamper: bool = False) -> tuple[dict, ArchiveServer]:
    server = ArchiveServer(tamper=tamper)
    namespace = {
        "__name__": "__notebook__",
        "os": os,
        "urllib": urllib,
        "Path": Path,
        "mp": mp,
        "display": lambda *args, **kwargs: None,
        "NETWORK_TIMEOUT_SECONDS": 30,
    }
    with tempfile.TemporaryDirectory() as scratch, mock.patch("urllib.request.urlopen", server):
        cwd = os.getcwd()
        os.chdir(scratch)  # Colab/Kaggle do not run from the repository root
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                exec(compile(with_selector(cell, selector), f"{NOTEBOOK.name}:data-cell", "exec"), namespace)
        finally:
            os.chdir(cwd)
    return namespace, server


class SampleRegistryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cell = load_data_cell()
        cls.registry = registry(cls.cell)

    def test_registry_lists_every_committed_archive(self) -> None:
        files = {cfg["file"] for cfg in self.registry.values()}
        committed = {path.name for path in SAMPLE_DIR.glob("*.zip")}
        self.assertEqual(files, committed, "registry files and committed archives differ")

    def test_registry_digests_match_committed_archives(self) -> None:
        for selector, cfg in self.registry.items():
            with self.subTest(selector=selector):
                self.assertIn("sha256", cfg, "registry entry carries no expected SHA-256")
                digest = hashlib.sha256((SAMPLE_DIR / cfg["file"]).read_bytes()).hexdigest()
                self.assertEqual(cfg["sha256"], digest)

    def test_every_variant_loads_without_network(self) -> None:
        for selector, cfg in self.registry.items():
            with self.subTest(selector=selector):
                namespace, server = run_cell(self.cell, selector)
                self.assertEqual(len(server.urls), 1, server.urls)
                self.assertTrue(server.urls[0].endswith("/" + cfg["file"]), server.urls[0])
                for fragment in FORBIDDEN_URL_FRAGMENTS:
                    self.assertNotIn(fragment, server.urls[0])
                self.assertEqual(namespace["TARGET_COLUMN"], cfg["target"])
                for name in ("train_data", "holdout_data", "test_data"):
                    frame = namespace[name]
                    self.assertGreater(len(frame), 0, name)
                    self.assertIn(cfg["target"], frame.columns, name)
                self.assertEqual(namespace.get("DATA_DIGEST"), cfg.get("sha256"))

    def test_tampered_archive_is_rejected(self) -> None:
        selector = next(iter(self.registry))
        with self.assertRaises(ValueError):
            run_cell(self.cell, selector, tamper=True)


if __name__ == "__main__":
    unittest.main()
