"""Validation-only checkpoint ranking and a configurable minimum training floor."""


def selection_score(metrics, primary="accuracy"):
    """Return a deterministic lexicographic key; preserve the original default."""
    priorities = {
        "accuracy": ("accuracy", "macro_f1"),
        "macro_f1": ("macro_f1", "macro_recall", "accuracy"),
        "macro_recall": ("macro_recall", "macro_f1", "accuracy"),
    }
    if primary not in priorities:
        raise ValueError(f"Unsupported validation selection metric: {primary}")
    return tuple(float(metrics[key]) for key in priorities[primary])


def selection_description(primary="accuracy"):
    descriptions = {
        "accuracy": "val accuracy, tie macro F1",
        "macro_f1": "val macro F1, tie macro recall, then accuracy",
        "macro_recall": "val macro recall, tie macro F1, then accuracy",
    }
    if primary not in descriptions:
        raise ValueError(f"Unsupported validation selection metric: {primary}")
    return descriptions[primary]


def should_stop(completed_epochs, stale, patience, minimum_epochs=0):
    if patience < 1 or minimum_epochs < 0:
        raise ValueError("Patience must be positive and minimum_epochs nonnegative")
    return completed_epochs >= minimum_epochs and stale >= patience
