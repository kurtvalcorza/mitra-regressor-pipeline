"""Checks for the uv isolated environment of both FreshRetailNet regression workshops (2026-10-03).

Nothing is installed into the notebook kernel: Section 0.1 only checks the runtime, and Section 5.2 builds each
foundation-model environment with a pinned uv from a hash-locked requirements file. The dynamic tests execute the
notebooks' own cell code with the subprocess layer replaced by a recorder. They are logic checks, not installs.
"""

from __future__ import annotations

import ast
import functools
import hashlib
import json
import os
import re
import shutil
import subprocess
import types
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
TUTORIALS = ROOT / "tutorials"
NOTEBOOKS = {
    "v1": TUTORIALS / "DIMER_FreshRetailNet_MultiModel_Regression_Workshop.ipynb",
    "v2": TUTORIALS / "DIMER_FreshRetailNet_MultiModel_Regression_Workshop_v2.ipynb",
}
STACKS = ("mitra", "tabdpt", "tabpfn", "tabicl")
MAX_LINE = 2000


@functools.cache
def _cells(edition: str) -> list[dict]:
    return json.loads(NOTEBOOKS[edition].read_text(encoding="utf-8"))["cells"]


def _code(edition: str, title: str) -> str:
    (source,) = ["".join(c["source"]) for c in _cells(edition) if "".join(c["source"]).startswith(f"# @title {title} ")]
    return source


def _literal(source: str, name: str):
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise KeyError(name)


def _lock(stack: str, suffix: str = "") -> str:
    return (TUTORIALS / f"requirements-workshop-{stack}{suffix}.lock.txt").read_text(encoding="utf-8")


def _requirements(text: str) -> list[tuple[str, list[str]]]:
    blocks = [block for block in re.split(r"\n(?=\S)", text) if block and not block.startswith("#")]
    return [(block.split(" ", 1)[0], re.findall(r"--hash=sha256:[0-9a-f]{64}", block)) for block in blocks]


@pytest.mark.parametrize("edition", NOTEBOOKS)
def test_nothing_is_installed_into_the_kernel_and_no_restart_is_needed(edition: str) -> None:
    for cell in _cells(edition):
        source = "".join(cell["source"])
        if cell["cell_type"] == "code":
            # `uv pip install --python <environment>` is allowed; `python -m pip` (into any interpreter) is not.
            assert not re.search(r"""["']-m["']\s*,\s*["']pip["']""", source), source[:60]
            assert not re.search(r"(?m)^\s*[%!]?\s*pip\s+install", source)
            assert not re.search(r"(?m)^\s*[%!]", source)
        assert "Restart session, then" not in source
        assert "install LightGBM if it is missing" not in source
    runtime_check = _code(edition, "0.1")
    assert "Nothing was installed" in runtime_check and "\"pip\"" not in runtime_check and "subprocess" not in runtime_check


@pytest.mark.parametrize("edition", NOTEBOOKS)
def test_no_cell_line_exceeds_2000_characters(edition: str) -> None:
    longest = max(len(line) for cell in _cells(edition) for line in "".join(cell["source"]).split("\n"))
    assert longest <= MAX_LINE


@pytest.mark.parametrize("stack", STACKS)
def test_each_stack_has_a_hash_locked_requirements_file(stack: str) -> None:
    text = _lock(stack)
    entries = _requirements(text)
    assert entries and all(re.fullmatch(r"[A-Za-z0-9_.\-]+==[^=\s]+", name) for name, _ in entries)
    assert all(hashes for _, hashes in entries), "every pinned requirement needs at least one SHA-256"
    pins = _literal(_code("v1", "5.2"), "MODEL_ENVIRONMENT_DEPENDENCIES")[stack]
    locked = {name.split("==")[0].lower().replace("_", "-").replace(".", "-"): name.split("==")[1] for name, _ in entries}
    if stack == "tabdpt":
        locked.update({n.split("==")[0]: n.split("==")[1] for n, _ in _requirements(_lock(stack, "-sdist"))})
    for pin in pins:
        name, version = pin.split("==")
        assert locked[re.sub(r"\[.*\]", "", name).lower().replace("_", "-").replace(".", "-")] == version, pin
        assert f"#   {pin}\n" in text, f"lock header must state the notebook pin {pin}"


def test_the_only_source_built_package_is_tabdpts_antlr_runtime() -> None:
    sources = sorted(path.name for path in TUTORIALS.glob("requirements-workshop-*-sdist.lock.txt"))
    assert sources == ["requirements-workshop-tabdpt-sdist.lock.txt"]
    assert [name for name, _ in _requirements(_lock("tabdpt", "-sdist"))] == ["antlr4-python3-runtime==4.9.3"]


@pytest.mark.parametrize("edition", NOTEBOOKS)
def test_notebook_carries_the_repository_locks_verbatim(edition: str) -> None:
    cell = _code(edition, "5.2")
    assert _literal(cell, "CARRIED_ENVIRONMENT_LOCKS") == {stack: _lock(stack) for stack in STACKS}
    assert _literal(cell, "CARRIED_SOURCE_BUILD_LOCKS") == {"tabdpt": _lock("tabdpt", "-sdist")}


def test_both_editions_share_the_runtime_and_environment_cells() -> None:
    for title in ("0.1", "0.2", "0.3", "5.2", "5.3", "5.4"):
        assert _code("v1", title) == _code("v2", title), title


# --- Dynamic: run the notebook's Section 5.2 cell with a recording subprocess layer ------------------------------


class _Recorder:
    PIPE = STDOUT = None
    CalledProcessError = subprocess.CalledProcessError

    def __init__(self):
        self.commands: list[list[str]] = []
        self.environments: list[dict] = []

    def run(self, command, **kwargs):
        return subprocess.CompletedProcess(command, 0, stdout="3.12.12\n", stderr="")

    def Popen(self, command, env=None, **kwargs):  # noqa: N802 - mirrors subprocess.Popen
        command = [str(part) for part in command]
        self.commands.append(command)
        self.environments.append(dict(env or {}))
        if command[1:3] == ["venv", "--managed-python"]:
            python = Path(command[-1]) / "bin" / "python"
            python.parent.mkdir(parents=True, exist_ok=True)
            python.write_text("")
        return types.SimpleNamespace(stdout=iter(()), wait=lambda: 0)


def _run_environment_cell(edition: str, tmp_path: Path, keys: list[str]) -> tuple[dict, _Recorder]:
    recorder = _Recorder()
    source = _code(edition, "5.2")
    fake_uv = tmp_path / "uv"
    assert source.count("uv_path=ensure_uv()") == 1
    source = source.replace("uv_path=ensure_uv()", f"uv_path=Path({str(fake_uv)!r})")
    specs = {"mitra": "mitra_icl", "tabdpt": "tabdpt_icl", "tabpfn": "tabpfn3_icl", "tabicl": "tabiclv2_icl"}
    namespace = {
        "os": os, "json": json, "hashlib": hashlib, "shutil": shutil, "subprocess": recorder, "Path": Path,
        "pd": pd, "display": lambda *a, **k: None,
        "fingerprint": lambda value: hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest(),
        "sha256_file": lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        "WORKSPACE_ROOT": tmp_path / "workspace", "SESSION_ROOT": tmp_path / "session", "SUPPORT_DIR": tmp_path / "support",
        "CONDITION_SPECS": {condition: {"environment": env} for env, condition in specs.items()},
        "selected_foundation_conditions": [specs[key] for key in keys],
        "REBUILD_MODEL_ENVIRONMENTS": False, "REQUIRE_ALL_SELECTED_MODELS": True,
    }
    os.environ["PYTHONSTARTUP"] = "/kernel/startup.py"
    try:
        exec(compile(source, "cell-5.2", "exec"), namespace)
    finally:
        os.environ.pop("PYTHONSTARTUP", None)
    return namespace, recorder


@pytest.mark.parametrize("edition", NOTEBOOKS)
def test_environments_are_built_with_pinned_uv_and_hash_locked_wheels_only(edition: str, tmp_path: Path) -> None:
    namespace, recorder = _run_environment_cell(edition, tmp_path, ["tabpfn", "tabdpt"])
    assert not namespace["ENVIRONMENT_ERRORS"]
    by_stack: dict[str, list[list[str]]] = {}
    for command in recorder.commands:
        stack = next((s for s in ("tabpfn", "tabdpt") if any(f"/{s}-" in part.replace("\\", "/") for part in command)), None)
        by_stack.setdefault(stack, []).append(command)
    for stack in ("tabpfn", "tabdpt"):
        commands = by_stack[stack]
        venv, install = commands[0], commands[1]
        assert venv[1:5] == ["venv", "--managed-python", "--python", "3.12.12"]
        assert install[1:3] == ["pip", "install"]
        for flag in ("--require-hashes", "--no-deps"):
            assert flag in install
        assert install[install.index("--only-binary") + 1] == ":all:"
        assert install[install.index("--index-url") + 1] == "https://pypi.org/simple"
        assert Path(install[-1]).read_text(encoding="utf-8") == _lock(stack)
        assert commands[-2][1:3] == ["pip", "check"]
        assert commands[-1][1] == "-c" and commands[-1][0].replace("\\", "/").endswith("/bin/python")
    source_builds = [c for c in by_stack["tabdpt"] if "--no-build-isolation" in c]
    assert len(source_builds) == 1 and "--require-hashes" in source_builds[0] and "--only-binary" not in source_builds[0]
    assert Path(source_builds[0][-1]).read_text(encoding="utf-8") == _lock("tabdpt", "-sdist")
    assert not any("--no-build-isolation" in c for c in by_stack["tabpfn"])


@pytest.mark.parametrize("edition", NOTEBOOKS)
def test_model_stages_run_with_the_environment_python_and_a_clean_child_environment(edition: str, tmp_path: Path) -> None:
    namespace, recorder = _run_environment_cell(edition, tmp_path, ["tabicl"])
    python = namespace["MODEL_ENVIRONMENT_PYTHONS"]["tabicl"]
    assert python.parent.name == "bin" and python.name == "python" and python.is_relative_to(tmp_path / "workspace")
    assert namespace["describe_environment"]("tabicl")["python"] == python
    assert namespace["describe_environment"]("tabicl")["runtime"]["lock_sha256"] == hashlib.sha256(
        _lock("tabicl").encode("utf-8")).hexdigest()
    for environment in recorder.environments:
        assert environment["MPLBACKEND"] == "Agg"
        assert environment["PYTHONPATH"] == str(namespace["CARRIED_SOURCE_ROOT"])
        assert "PYTHONSTARTUP" not in environment and "PYTHONHOME" not in environment
    # The carried experiment engine launches every validation and frozen-test run with that interpreter.
    execution = _literal(_code(edition, "0.3"), "EXECUTION_SOURCE")
    assert execution.count('self.execute_command([str(environment["python"]),str(self.runner)') == 2


@pytest.mark.parametrize("edition", NOTEBOOKS)
def test_runtime_check_refuses_non_linux_kernels(edition: str, monkeypatch) -> None:
    import platform

    source = _code(edition, "0.1")
    monkeypatch.setattr(platform, "system", lambda: "Windows")
    with pytest.raises(RuntimeError, match="Linux x86_64"):
        exec(compile(source, "cell-0.1", "exec"), {})
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    monkeypatch.setattr(platform, "machine", lambda: "x86_64")
    monkeypatch.setattr("importlib.util.find_spec", lambda name: None if name == "lightgbm" else object())
    with pytest.raises(RuntimeError, match="lightgbm"):
        exec(compile(source, "cell-0.1", "exec"), {})
