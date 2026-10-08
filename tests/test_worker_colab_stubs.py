"""The isolated worker's google.colab shims are well-formed modules (NOTEBOOK_SPEC 2.3 ENV15/ENV16).

Libraries call importlib.util.find_spec("google.colab") (accelerate does); on a stub whose __spec__ is None that raises
ValueError on Colab only. Runs the worker's own shim source from each generated notebook with DIMER_KERNEL_IS_COLAB=1.
CI's dependencies only.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = [ROOT / "tutorials" / "mitra_regressor_colab.ipynb", ROOT / "tutorials" / "mitra_regressor_predictor_inference_colab.ipynb"]


def _shim(path: Path) -> str:
    cells = json.loads(path.read_text(encoding="utf-8"))["cells"]
    router = [c["source"] for c in cells if c["cell_type"] == "code" and '_WORKER_SOURCE = r"""' in c["source"]]
    assert len(router) == 1
    worker = router[0][router[0].index('_WORKER_SOURCE = r"""') + len('_WORKER_SOURCE = r"""') :]
    worker = worker[: worker.index('"""')]
    start = worker.index('if os.environ.get("DIMER_KERNEL_IS_COLAB") == "1":')
    return worker[start : worker.index('_main = types.ModuleType("__main__")', start)]


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.stem)
@pytest.mark.parametrize("real_google", [False, True])
def test_worker_colab_stubs_have_specs(path: Path, real_google: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    shim = _shim(path)
    names = ("google", "google.colab", "google.colab.files")
    saved = {n: sys.modules[n] for n in names if n in sys.modules}
    fake_google = types.ModuleType("google")
    fake_google.__path__ = []
    try:
        for n in names:
            sys.modules.pop(n, None)
        sys.modules["google"] = fake_google if real_google else None
        monkeypatch.setenv("DIMER_KERNEL_IS_COLAB", "1")
        exec(compile(shim, "worker-colab-shim", "exec"), {"os": os, "sys": sys, "types": types, "_send": None, "_recv": None})
        for n in ("google.colab", "google.colab.files"):
            spec = importlib.util.find_spec(n)  # raised ValueError: google.colab.__spec__ is None before the fix
            assert spec is not None and spec.name == n
        assert sys.modules["google.colab"].__path__ == [] and callable(sys.modules["google.colab.files"].upload)
        if not real_google:
            assert importlib.util.find_spec("google") is not None
    finally:
        for n in names:
            sys.modules.pop(n, None)
        sys.modules.update(saved)


def test_shim_without_the_colab_flag_registers_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DIMER_KERNEL_IS_COLAB", raising=False)
    before = {n: sys.modules.get(n) for n in ("google.colab", "google.colab.files")}
    exec(compile(_shim(NOTEBOOKS[0]), "worker-colab-shim", "exec"), {"os": os, "sys": sys, "types": types, "_send": None, "_recv": None})
    assert {n: sys.modules.get(n) for n in before} == before
