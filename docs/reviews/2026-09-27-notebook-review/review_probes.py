"""Reduced offline probes for the pinned FreshRetailNet regression notebook.

Scope: manually transcribed selectors/scorer/discretization from sections 8.0,
8.3, and 9.1. Synthetic data replaces notebook results. This does NOT execute
Session.freeze, model adapters, a Colab kernel, or the full notebook. Deliberate
bad-path outcomes reproduce defects; positive controls detect harness mistakes.
"""
from __future__ import annotations
import json
import platform
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import balanced_accuracy_score, mean_absolute_error, r2_score

COMMIT = 'dd7c8ecd49ad1c7de4cae5875152a71c40f431d7'
BLOB = '026741c58e5a9557d14320f72239afc411f2351a'

# Section 8.3, transcribed with spacing expanded only.
def score_for(metric, truth, prediction):
    error = truth - prediction
    if metric == 'mae':
        return float(np.mean(np.abs(error)))
    if metric == 'rmse':
        return float(np.sqrt(np.mean(error ** 2)))
    if metric == 'r2':
        return float(1 - np.sum(error ** 2) / np.sum((truth - truth.mean()) ** 2))
    raise ValueError(metric)


def reference_key_for(primary_metric, baseline_validation_results, bootstrap_predictions):
    higher_is_better = primary_metric == 'r2'
    classical_ranking = baseline_validation_results.sort_values(
        primary_metric, ascending=not higher_is_better
    ).model_key
    return next(key for key in classical_ranking if key in bootstrap_predictions)


def band_test(train_targets, eval_targets, predictions):
    # Section 9.1's actual quantile/digitize/BA computation, no model invocation.
    band_labels = np.array(['low', 'mid', 'high'])
    edges = pd.Series(train_targets).quantile([1 / 3, 2 / 3]).to_numpy(dtype=float)
    truth = band_labels[np.digitize(np.asarray(eval_targets, dtype=float), edges)]
    pred = band_labels[np.digitize(np.asarray(predictions, dtype=float), edges)]
    return {
        'edges': edges.tolist(), 'truth_bands': truth.tolist(),
        'prediction_bands': pred.tolist(),
        'balanced_accuracy': float(balanced_accuracy_score(truth, pred)),
        'observed_true_classes': sorted(set(truth)),
    }


def main():
    results = []
    ranking = pd.DataFrame({
        'model_key': ['lightgbm', 'random_forest'], 'mae': [0.373, 0.380],
        'rmse': [0.700, 0.720], 'r2': [0.72, 0.70],
    })
    # These names pass the visible Section 8.0 successful-key membership check.
    eligible = ['lightgbm', 'random_forest', 'tabiclv2_icl', 'mitra_icl']
    selected = ['tabiclv2_icl', 'mitra_icl']
    assert not any(k not in eligible for k in selected)
    try:
        reference_key_for('mae', ranking, {k: object() for k in selected})
    except StopIteration:
        results.append({'id': 'P1', 'finding': 'REG-01',
                        'outcome': 'reproduced',
                        'observed': 'StopIteration when the frozen prediction set contains foundation models only',
                        'limit': 'Visible selection membership and bootstrap selector only; Session.freeze not executed'})
    else:
        raise AssertionError('P1 should reproduce StopIteration')

    ref = reference_key_for('mae', ranking, {'lightgbm': object(), 'tabiclv2_icl': object()})
    assert ref == 'lightgbm'
    results.append({'id': 'P2', 'outcome': 'positive control passed', 'reference_key': ref})

    truth = np.array([0.0, 1.0])
    prediction = np.array([0.1, 0.9])
    full = score_for('r2', truth, prediction)
    assert np.isclose(full, r2_score(truth, prediction)) and np.isfinite(full)
    rng = np.random.default_rng(0)
    draws = rng.integers(0, len(truth), size=(1000, len(truth)))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        values = np.array([score_for('r2', truth[r], prediction[r]) for r in draws])
        interval = np.percentile(values, [2.5, 97.5])
        gains = values - values
    nonfinite = int((~np.isfinite(values)).sum())
    assert nonfinite > 0 and not np.isfinite(interval).all()
    results.append({'id': 'P3', 'finding': 'REG-02', 'outcome': 'reproduced',
                    'full_test_r2': full, 'bootstrap_draws': len(draws),
                    'nonfinite_r2_draws': nonfinite,
                    'interval_as_strings': [str(v) for v in interval],
                    'nonfinite_self_difference_draws': int((~np.isfinite(gains)).sum()),
                    'scope': 'R2 selected with a two-row nonconstant test target; not the default MAE sample'})

    mae = np.array([score_for('mae', truth[r], prediction[r]) for r in draws])
    rmse = np.array([score_for('rmse', truth[r], prediction[r]) for r in draws])
    assert np.isfinite(mae).all() and np.isfinite(rmse).all()
    assert np.isclose(score_for('mae', truth, prediction), mean_absolute_error(truth, prediction))
    results.append({'id': 'P4', 'outcome': 'positive control passed',
                    'mae_rmse_draws_all_finite': True})

    collapsed = band_test([0.0] * 70 + [1.0] * 20 + [2.0] * 10,
                          [0.0, 1.0, 2.0], [1.0, 1.0, 1.0])
    assert collapsed['edges'] == [0.0, 0.0]
    assert collapsed['balanced_accuracy'] == 1.0
    assert collapsed['observed_true_classes'] == ['high']
    results.append({'id': 'P5', 'finding': 'REG-03', 'outcome': 'reproduced',
                    'train_rows': 100, 'train_distinct_numeric_targets': 3, **collapsed,
                    'interpretation': 'Correct metric for a one-class derived task, not evidence of successful three-band classification'})

    ordinary = band_test([0.0, 1.0, 2.0] * 40, [0.0, 1.0, 2.0], [0.0, 0.0, 0.0])
    assert len(ordinary['observed_true_classes']) == 3
    assert np.isclose(ordinary['balanced_accuracy'], 1/3)
    results.append({'id': 'P6', 'outcome': 'positive control passed', **ordinary})

    payload = {
        'notebook_commit': COMMIT, 'notebook_blob': BLOB,
        'execution_scope': 'Reduced, transcribed logic with synthetic fixtures; no full notebook or model execution',
        'runtime': {'python': platform.python_version(), 'numpy': np.__version__,
                    'pandas': pd.__version__, 'scikit_learn': sklearn.__version__},
        'checks': results,
    }
    output = Path(__file__).with_name('probe_results.json')
    output.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(payload, indent=2, allow_nan=False))

if __name__ == '__main__':
    main()
