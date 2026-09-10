from __future__ import annotations

import json
from pathlib import Path

NOTEBOOK = Path("tutorials/mitra_regressor_colab.ipynb")
README = Path("tutorials/README.md")

nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
cell = next(
    c
    for c in nb["cells"]
    if c.get("cell_type") == "code"
    and 'DATA_SOURCE = "Sample dataset (FreshRetailNet)"' in "".join(c.get("source", []))
)
src = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]

old_line = (
    'DATA_SOURCE = "Sample dataset (FreshRetailNet)"  # @param '
    '["Sample dataset (FreshRetailNet)", "Upload CSV", "Upload pre-split train/val/test"]'
)
new_line = (
    'DATA_SOURCE = "Sample: FreshRetailNet (temporal demand)"  # @param '
    '["Sample: FreshRetailNet (temporal demand)", '
    '"Sample: Insurance Charges (medical cost)", '
    '"Sample: Ames Housing (home valuation)", '
    '"Sample dataset (FreshRetailNet)", "Upload CSV", '
    '"Upload pre-split train/val/test"]'
)
if old_line not in src:
    raise SystemExit("DATA_SOURCE selector anchor not found")
src = src.replace(old_line, new_line, 1)

old_block = """SAMPLE_REVISION = \"5625a9eeca94b8c72b9ad1ec78d07ecbaa720903\"
SAMPLE_URL = (
    \"https://raw.githubusercontent.com/kurtvalcorza/mitra-regressor-pipeline/\"
    f\"{SAMPLE_REVISION}/examples/sample-data/freshretailnet-h7.zip\"
)
SAMPLE_CARD_URL = (
    \"https://github.com/kurtvalcorza/mitra-regressor-pipeline/blob/\"
    f\"{SAMPLE_REVISION}/examples/sample-data/DATASET_CARD.md\"
)
"""
new_block = """SAMPLE_REVISION = \"f02e0c38ce835d6b85b5a6f072d232f3cd306f54\"
SAMPLE_BASE_URL = (
    \"https://raw.githubusercontent.com/kurtvalcorza/mitra-regressor-pipeline/\"
    f\"{SAMPLE_REVISION}/examples/sample-data\"
)
SAMPLE_CARD_URL = (
    \"https://github.com/kurtvalcorza/mitra-regressor-pipeline/blob/\"
    f\"{SAMPLE_REVISION}/examples/sample-data/DATASET_CARD.md\"
)
SAMPLE_CONFIGS = {
    \"Sample: FreshRetailNet (temporal demand)\": {
        \"file\": \"freshretailnet-h7.zip\", \"target\": \"target\"
    },
    \"Sample dataset (FreshRetailNet)\": {
        \"file\": \"freshretailnet-h7.zip\", \"target\": \"target\"
    },
    \"Sample: Insurance Charges (medical cost)\": {
        \"file\": \"insurance-medical-charges.zip\", \"target\": \"charges\"
    },
    \"Sample: Ames Housing (home valuation)\": {
        \"file\": \"ames-housing.zip\", \"target\": \"SalePrice\"
    },
}
"""
if old_block not in src:
    raise SystemExit("sample URL anchor not found")
src = src.replace(old_block, new_block, 1)
src = src.replace(
    'using_sample = DATA_SOURCE == "Sample dataset (FreshRetailNet)"',
    "using_sample = DATA_SOURCE in SAMPLE_CONFIGS",
    1,
)
src = src.replace(
    '    "Sample dataset (FreshRetailNet)",\n    "Upload pre-split train/val/test",',
    '    *SAMPLE_CONFIGS,\n    "Upload pre-split train/val/test",',
    1,
)
anchor = """if using_sample:
    with urllib.request.urlopen(SAMPLE_URL, timeout=NETWORK_TIMEOUT_SECONDS) as response:
"""
replacement = """if using_sample:
    sample_cfg = SAMPLE_CONFIGS[DATA_SOURCE]
    TARGET_COLUMN = sample_cfg[\"target\"]
    SAMPLE_URL = f\"{SAMPLE_BASE_URL}/{sample_cfg['file']}\"
    print(\"Sample:\", DATA_SOURCE)
    print(\"Target:\", TARGET_COLUMN)
    with urllib.request.urlopen(SAMPLE_URL, timeout=NETWORK_TIMEOUT_SECONDS) as response:
"""
if anchor not in src:
    raise SystemExit("sample load anchor not found")
src = src.replace(anchor, replacement, 1)
cell["source"] = src
NOTEBOOK.write_text(json.dumps(nb, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

text = README.read_text(encoding="utf-8")
if "## Sample portfolio\n" not in text:
    text = text.rstrip() + """

## Sample portfolio

The E2E notebook preserves the original FreshRetailNet temporal-demand sample and also exposes the sample portfolio from PR #19: Insurance Medical Charges (`charges`) and Ames Housing (`SalePrice`). These archives and their provenance/license details live under `examples/sample-data/`. The older `Sample dataset (FreshRetailNet)` selector remains accepted for backward compatibility. Sample metrics remain tutorial/sanity evidence, not benchmark claims.
"""
    README.write_text(text, encoding="utf-8")
