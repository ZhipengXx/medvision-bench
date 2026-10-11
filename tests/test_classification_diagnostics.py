"""Validate vectorized paired-statistics helpers against independent definitions."""
import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import confusion_matrix, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.diagnose_classification import auc_batches, confusion_batches, holm, mcnemar, scores


def test_bootstrap_auc_handles_ties_and_repeated_images():
    y = np.tile(np.arange(7), 3)
    rng = np.random.default_rng(21)
    probability = rng.integers(1, 5, (len(y), 7)).astype(float)
    probability /= probability.sum(1, keepdims=True)
    weights = np.stack([np.ones(len(y)), np.tile([1, 2, 3], 7)])
    actual = auc_batches(weights, y, probability)
    for b, sample_weight in enumerate(weights):
        indices = np.repeat(np.arange(len(y)), sample_weight.astype(int))
        expected = np.mean([roc_auc_score(y[indices] == k, probability[indices, k]) for k in range(7)])
        assert actual[b] == pytest.approx(expected)


def test_bootstrap_confusion_matches_explicit_resampling():
    y = np.tile(np.arange(7), 2)
    pred = np.array([0, 1, 2, 0, 0, 6, 6, 1, 1, 2, 3, 4, 5, 6])
    weights = np.arange(1, len(y) + 1)[None, :]
    indices = np.repeat(np.arange(len(y)), weights[0])
    cm = confusion_batches(weights, y, pred)
    np.testing.assert_array_equal(cm[0], confusion_matrix(y[indices], pred[indices], labels=np.arange(7)))
    assert scores(cm)['accuracy'][0] == pytest.approx(np.mean(y[indices] == pred[indices]))


def test_exact_mcnemar_and_holm_families():
    y = np.zeros(5, dtype=int)
    test = mcnemar(y, np.ones(5, dtype=int), y)
    assert test['baseline_wrong_other_correct'] == 5
    assert test['exact_two_sided_p'] == pytest.approx(0.0625)
    assert holm([0.04, 0.001, 0.03]) == pytest.approx([0.06, 0.003, 0.06])
