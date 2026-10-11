"""Check real selection/early-stop behaviors without loading models or test data."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.classification.selection import selection_score, should_stop


def test_legacy_accuracy_rule_is_preserved():
    majority = dict(accuracy=0.9, macro_f1=0.6, macro_recall=0.5)
    balanced = dict(accuracy=0.8, macro_f1=0.8, macro_recall=0.8)
    assert selection_score(majority) == (0.9, 0.6)
    assert selection_score(majority) > selection_score(balanced)
    assert selection_score(balanced, 'macro_f1') > selection_score(majority, 'macro_f1')


def test_f1_ties_use_recall_then_accuracy():
    a = dict(accuracy=0.9, macro_f1=0.8, macro_recall=0.6)
    b = dict(accuracy=0.8, macro_f1=0.8, macro_recall=0.7)
    assert selection_score(b, 'macro_f1') > selection_score(a, 'macro_f1')
    assert selection_score({**b, 'accuracy': 0.81}, 'macro_f1') > selection_score(b, 'macro_f1')


def test_minimum_epochs_prevents_premature_stopping():
    assert should_stop(12, 10, 10)  # Original WCE stopping condition.
    assert not should_stop(12, 10, 10, minimum_epochs=20)
    assert not should_stop(19, 17, 10, minimum_epochs=20)
    assert should_stop(20, 18, 10, minimum_epochs=20)
    assert not should_stop(20, 0, 10, minimum_epochs=20)  # New best resets stale.


def test_invalid_protocol_is_rejected():
    with pytest.raises(ValueError):
        selection_score({}, 'test_accuracy')
    with pytest.raises(ValueError):
        should_stop(20, 10, 0)


def test_sensitivity_freeze_uses_f1_and_keeps_first_exact_tie():
    from scripts.freeze_sensitivity import best_history
    history = [
        dict(epoch=1, val_accuracy=0.9, val_macro_f1=0.6, val_macro_recall=0.5),
        dict(epoch=2, val_accuracy=0.8, val_macro_f1=0.8, val_macro_recall=0.7),
        dict(epoch=3, val_accuracy=0.8, val_macro_f1=0.8, val_macro_recall=0.7),
    ]
    assert best_history(history)['epoch'] == 2
