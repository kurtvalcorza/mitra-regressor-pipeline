from __future__ import annotations

import json
from pathlib import Path

path = Path("tutorials/mitra_regressor_colab.ipynb")
nb = json.loads(path.read_text(encoding="utf-8"))

for cell in nb["cells"]:
    src = cell.get("source", "")
    text = "".join(src) if isinstance(src, list) else src
    if "SAMPLE_CONFIGS = {" not in text or "drop_columns =" not in text:
        continue
    old = '''drop_columns = [c.strip() for c in DROP_COLUMNS.split(",") if c.strip() and c.strip() != TARGET_COLUMN]
using_sample = DATA_SOURCE in SAMPLE_CONFIGS
using_presplit = DATA_SOURCE in {
    *SAMPLE_CONFIGS,
    "Upload pre-split train/val/test",
}
'''
    new = '''using_sample = DATA_SOURCE in SAMPLE_CONFIGS
if using_sample:
    # Resolve the effective target before interpreting DROP_COLUMNS. Otherwise a
    # non-default sample can accidentally request its real target as a dropped feature.
    TARGET_COLUMN = SAMPLE_CONFIGS[DATA_SOURCE]["target"]

drop_columns = [c.strip() for c in DROP_COLUMNS.split(",") if c.strip() and c.strip() != TARGET_COLUMN]
using_presplit = DATA_SOURCE in {
    *SAMPLE_CONFIGS,
    "Upload pre-split train/val/test",
}
'''
    if old not in text:
        raise RuntimeError("expected sample/drop ordering block not found")
    text = text.replace(old, new, 1)
    cell["source"] = text
    break
else:
    raise RuntimeError("tutorial data cell not found")

path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
