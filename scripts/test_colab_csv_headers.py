"""Exercise real regressor notebook CSV parsing without Colab or fitting."""

import ast
import csv
import io
import json
from pathlib import Path
import types
import unittest
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MAIN_NOTEBOOK = "mitra_regressor_colab.ipynb"
NOTEBOOKS = (
    MAIN_NOTEBOOK,
    "mitra_regressor_predictor_inference_colab.ipynb",
)


def notebook_code(name):
    notebook = json.loads((ROOT / "tutorials" / name).read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        if cell.get("cell_type") != "code":
            continue
        source = cell.get("source", "")
        yield "".join(source) if isinstance(source, list) else source


def inference_cell(name):
    for source in notebook_code(name):
        if "read_inference_csv" in source and "FEATURE_COLUMNS" in source:
            return source
    raise AssertionError(f"Could not locate inference CSV cell in {name}")


def training_reader():
    for source in notebook_code(MAIN_NOTEBOOK):
        if "def read_csv_payload" not in source:
            continue
        tree = ast.parse(source)
        function = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "read_csv_payload"
        )
        module = ast.Module(body=[function], type_ignores=[])
        namespace = {"csv": csv, "io": io, "pd": pd}
        exec(compile(ast.fix_missing_locations(module), MAIN_NOTEBOOK, "exec"), namespace)
        return namespace["read_csv_payload"]
    raise AssertionError("Could not locate read_csv_payload in the main notebook")


def read_input_cell(name, payload, features):
    source = inference_cell(name)
    tree = ast.parse(source)
    if "RUN_NEW_DATA_INFERENCE" in source:
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "RUN_NEW_DATA_INFERENCE"
                for t in node.targets
            ):
                node.value = ast.Constant(True)
            if isinstance(node, ast.If) and isinstance(node.test, ast.Name) and node.test.id == "RUN_NEW_DATA_INFERENCE":
                end = next(
                    i for i, stmt in enumerate(node.body)
                    if isinstance(stmt, ast.Assign) and any(
                        isinstance(t, ast.Name) and t.id == "X" for t in stmt.targets
                    )
                )
                node.body = node.body[: end + 1]

    files = types.SimpleNamespace(upload=lambda: {"input.csv": payload})
    colab = types.ModuleType("google.colab")
    colab.files = files
    namespace = {
        "pd": pd,
        "files": files,
        "FEATURE_COLUMNS": features,
        "FIT_RUN_COMPLETED": True,
        "baseline_predictor": object(),
        "display": lambda *args: None,
        "print": lambda *args: None,
    }
    with patch.dict("sys.modules", {"google.colab": colab}):
        exec(compile(ast.fix_missing_locations(tree), name, "exec"), namespace)
    return namespace["X"]


DUPLICATE_CASES = (
    (b"amount,amount\n1,999\n", ["amount"]),
    (b'"sale,amount","sale,amount"\n1,999\n', ["sale,amount"]),
    (b'\xef\xbb\xbfamount,"amount"\r\n1,999\r\n', ["amount"]),
    (b'\namount,amount\n1,999\n', ["amount"]),
)


class CsvHeaderTests(unittest.TestCase):
    def test_duplicate_inference_headers_rejected(self):
        for name in NOTEBOOKS:
            for payload, features in DUPLICATE_CASES:
                with self.subTest(notebook=name, payload=payload):
                    with self.assertRaisesRegex(ValueError, "duplicate column names"):
                        read_input_cell(name, payload, features)

    def test_duplicate_training_headers_rejected(self):
        reader = training_reader()
        for payload, _ in DUPLICATE_CASES:
            with self.subTest(payload=payload):
                with self.assertRaisesRegex(ValueError, "duplicate column names"):
                    reader(payload, "uploaded CSV")

    def test_valid_headers_and_feature_order_preserved(self):
        for name in NOTEBOOKS:
            for payload in (
                b'amount.1,"sale,amount",amount\n7,9,1\n',
                b'\xef\xbb\xbfamount.1,"sale,amount",amount\r\n7,9,1\r\n',
            ):
                with self.subTest(notebook=name, payload=payload):
                    frame = read_input_cell(name, payload, ["amount", "sale,amount"])
                    self.assertEqual(list(frame.columns), ["amount", "sale,amount"])
                    self.assertEqual(frame.to_dict("list"), {"amount": [1], "sale,amount": [9]})


if __name__ == "__main__":
    unittest.main()
