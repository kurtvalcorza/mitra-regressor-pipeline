#!/usr/bin/env python3
"""One-shot migration helper: bound the literal default sample path for practical CPU execution."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "tutorials" / "mitra_regressor_colab.ipynb"
README = ROOT / "tutorials" / "README.md"
VALIDATOR = ROOT / "scripts" / "validate_colab_tutorial.py"


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if old not in text:
        raise RuntimeError(f"Could not locate expected {label} text")
    if text.count(old) != 1:
        raise RuntimeError(f"Expected exactly one {label} occurrence, found {text.count(old)}")
    return text.replace(old, new, 1)


payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
for cell in payload["cells"]:
    source = cell.get("source", "")
    source = "".join(source) if isinstance(source, list) else str(source)

    if cell.get("cell_type") == "markdown" and source.startswith("## 3. Load and validate sample or BYOD data"):
        source = replace_once(
            source,
            "The default sample is the pinned FreshRetailNet-derived regression package. Its provided `train.csv`, `val.csv`, and `test.csv` partitions are preserved. They are a purged chronological split with an embargo and are tutorial/sanity data, **not benchmark evidence**.",
            "The default sample is the pinned FreshRetailNet-derived regression package. Its provided `train.csv`, `val.csv`, and `test.csv` partition membership is preserved. They are a purged chronological split with an embargo and are tutorial/sanity data, **not benchmark evidence**. To keep the literal CPU-default path practical for an interactive tutorial and for clean release execution, the notebook deterministically draws a bounded smoke subset from within those already-separated partitions: 512 training rows and 256 rows from each evaluation split. Increase the two sample-row form values to use more or all rows; BYOD paths are unaffected by this smoke-only cap.",
            "step-3 sample description",
        )
        cell["source"] = source
        continue

    if cell.get("cell_type") == "code" and "DATA_SOURCE = \"Sample dataset (FreshRetailNet)\"" in source:
        source = replace_once(
            source,
            "SEED = 42                                        # @param {type:\"integer\"}\n\nSAMPLE_REVISION",
            "SEED = 42                                        # @param {type:\"integer\"}\nSAMPLE_TRAIN_ROWS = 512                         # @param {type:\"integer\"}\nSAMPLE_EVAL_ROWS = 256                          # @param {type:\"integer\"}\n\nSAMPLE_REVISION",
            "sample row controls",
        )
        source = replace_once(
            source,
            "        test_data = mp.read_csv_bytes(zf.read(names[\"test.csv\"]), \"test.csv\")\n    TARGET_COLUMN = \"target\"",
            "        test_data = mp.read_csv_bytes(zf.read(names[\"test.csv\"]), \"test.csv\")\n\n    def deterministic_smoke_subset(frame, rows, seed):\n        if rows <= 0:\n            raise ValueError(\"Sample row controls must be positive integers.\")\n        if len(frame) <= rows:\n            return frame.reset_index(drop=True)\n        return frame.sample(n=rows, random_state=seed).sort_index().reset_index(drop=True)\n\n    train_data = deterministic_smoke_subset(train_data, SAMPLE_TRAIN_ROWS, SEED)\n    holdout_data = deterministic_smoke_subset(holdout_data, SAMPLE_EVAL_ROWS, SEED + 1)\n    test_data = deterministic_smoke_subset(test_data, SAMPLE_EVAL_ROWS, SEED + 2)\n    print(\"Default smoke subset:\", {\"train\": len(train_data), \"holdout\": len(holdout_data), \"test\": len(test_data)})\n    TARGET_COLUMN = \"target\"",
            "sample subset application",
        )
        cell["source"] = source

NOTEBOOK.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

readme = README.read_text(encoding="utf-8")
readme = replace_once(
    readme,
    "The provided sample partitions are preserved. A BYOD single-CSV path uses a deterministic random holdout and explicitly assumes approximately IID rows.",
    "The provided sample partition membership is preserved. The literal default CPU tutorial deterministically uses 512 training rows and 256 rows from each evaluation partition as a bounded smoke subset; users can raise the notebook form controls to use more or all rows from the pinned convenience sample. This smoke-only cap does not apply to BYOD. A BYOD single-CSV path uses a deterministic random holdout and explicitly assumes approximately IID rows.",
    "README data/evaluation paragraph",
)
README.write_text(readme, encoding="utf-8")

validator = VALIDATOR.read_text(encoding="utf-8")
validator = replace_once(
    validator,
    '            "tutorial_metrics.json",\n            "If every default-path cell ran successfully",',
    '            "tutorial_metrics.json",\n            "SAMPLE_TRAIN_ROWS",\n            "SAMPLE_EVAL_ROWS",\n            "Default smoke subset:",\n            "If every default-path cell ran successfully",',
    "validator smoke markers",
)
validator = replace_once(
    validator,
    '    require(top_level_literal(code_cells, "RUN_FINE_TUNING", False),\n            "main notebook: RUN_FINE_TUNING must default False")',
    '    require(top_level_literal(code_cells, "SAMPLE_TRAIN_ROWS", 512),\n            "main notebook: SAMPLE_TRAIN_ROWS must default 512")\n    require(top_level_literal(code_cells, "SAMPLE_EVAL_ROWS", 256),\n            "main notebook: SAMPLE_EVAL_ROWS must default 256")\n    require(top_level_literal(code_cells, "RUN_FINE_TUNING", False),\n            "main notebook: RUN_FINE_TUNING must default False")',
    "validator sample literals",
)
validator = replace_once(
    validator,
    '            SAMPLE_REVISION,\n        ),',
    '            SAMPLE_REVISION,\n            "512 training rows",\n            "256 rows from each evaluation partition",\n        ),',
    "README smoke markers",
)
VALIDATOR.write_text(validator, encoding="utf-8")

print("Bounded default sample migration staged successfully")
