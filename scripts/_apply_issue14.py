#!/usr/bin/env python3
import json
from pathlib import Path

path = Path('tutorials/mitra_regressor_colab.ipynb')
nb = json.loads(path.read_text(encoding='utf-8'))
replacements = {
    "        train_data = pd.read_csv(z.open(names['train.csv']))\n": "        train_data = read_csv_payload(z.read(names['train.csv']), 'train.csv')\n",
    "        holdout_data = pd.read_csv(z.open(names['val.csv']))\n": "        holdout_data = read_csv_payload(z.read(names['val.csv']), 'val.csv')\n",
    "        test_data = pd.read_csv(z.open(names['test.csv']))\n": "        test_data = read_csv_payload(z.read(names['test.csv']), 'test.csv')\n",
}
counts = {old: 0 for old in replacements}
for cell in nb['cells']:
    if cell.get('cell_type') != 'code':
        continue
    source = cell.get('source', [])
    text = ''.join(source) if isinstance(source, list) else str(source)
    for old, new in replacements.items():
        if old in text:
            counts[old] += text.count(old)
            text = text.replace(old, new)
    cell['source'] = text.splitlines(keepends=True)

missing = [old.strip() for old, count in counts.items() if count != 1]
if missing:
    raise SystemExit(f'Expected exactly one occurrence for each ZIP reader; mismatches: {missing}')
path.write_text(json.dumps(nb, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
