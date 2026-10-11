"""Audit frozen classification predictions without fitting or changing any model.

Paired bootstrap resamples images within each ground-truth class, preserving the
observed test class mix. Its intervals describe image-sampling uncertainty for
these fixed fitted models; they do not measure seed or patient-level uncertainty.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]
RUNS = ["cls_r50_ce", "cls_r50_wce", "cls_r50_focal"]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_predictions(path):
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    ids = [r["image_id"] for r in rows]
    y = np.array([int(r["gt"]) for r in rows])
    pred = np.array([int(r["pred"]) for r in rows])
    prob = np.array([[float(r[f"p_{c}"]) for c in CLASSES] for r in rows])
    if len(ids) != len(set(ids)) or not len(ids):
        raise ValueError(f"Nonunique or empty image IDs: {path}")
    if not np.isin(y, np.arange(7)).all() or not np.isfinite(prob).all():
        raise ValueError(f"Invalid labels/probabilities: {path}")
    if (prob < 0).any() or not np.allclose(prob.sum(1), 1, atol=1e-5):
        raise ValueError(f"Probabilities are not normalized: {path}")
    if not np.array_equal(pred, prob.argmax(1)):
        raise ValueError(f"Saved argmax mismatch: {path}")
    return ids, y, pred, prob


def confusion_batches(weights, y, pred):
    cells = y * 7 + pred
    order = np.argsort(cells, kind="stable")
    unique, starts = np.unique(cells[order], return_index=True)
    result = np.zeros((len(weights), 49), dtype=float)
    result[:, unique] = np.add.reduceat(weights[:, order], starts, axis=1)
    return result.reshape(-1, 7, 7)


def scores(cm):
    true_positive = np.diagonal(cm, axis1=1, axis2=2)
    support, assigned = cm.sum(2), cm.sum(1)
    precision = np.divide(true_positive, assigned, out=np.zeros_like(true_positive), where=assigned > 0)
    recall = np.divide(true_positive, support, out=np.zeros_like(true_positive), where=support > 0)
    f1 = np.divide(2 * true_positive, support + assigned, out=np.zeros_like(true_positive), where=support + assigned > 0)
    n = support.sum(1)
    result = {
        "accuracy": true_positive.sum(1) / n,
        "macro_precision": precision.mean(1),
        "macro_recall": recall.mean(1),
        "macro_f1": f1.mean(1),
        "weighted_precision": (precision * support).sum(1) / n,
        "weighted_recall": (recall * support).sum(1) / n,
        "weighted_f1": (f1 * support).sum(1) / n,
    }
    for k, name in enumerate(CLASSES):
        result[f"recall_{name}"] = recall[:, k]
    return result


def auc_batches(weights, y, prob):
    """Exact weighted Mann-Whitney AUROC, with half credit for tied scores."""
    auc = []
    for k in range(7):
        order = np.argsort(prob[:, k], kind="stable")
        _, starts = np.unique(prob[order, k], return_index=True)
        positive = (y[order] == k)[None, :]
        weighted = weights[:, order]
        pos = np.add.reduceat(weighted * positive, starts, axis=1)
        neg = np.add.reduceat(weighted * ~positive, starts, axis=1)
        numerator = (pos * (np.cumsum(neg, axis=1) - 0.5 * neg)).sum(1)
        auc.append(numerator / (pos.sum(1) * neg.sum(1)))
    return np.mean(auc, axis=0)


def interval(values):
    return [float(v) for v in np.quantile(values, [0.025, 0.975])]


def holm(pvalues):
    order = np.argsort(pvalues)
    adjusted = np.empty(len(pvalues))
    previous = 0.0
    for i, k in enumerate(order):
        previous = max(previous, min(1.0, (len(pvalues) - i) * pvalues[k]))
        adjusted[k] = previous
    return adjusted.tolist()


def mcnemar(y, baseline, other):
    first, second = baseline == y, other == y
    gains = int((~first & second).sum())
    losses = int((first & ~second).sum())
    return dict(baseline_wrong_other_correct=gains,
                baseline_correct_other_wrong=losses,
                discordant=gains + losses,
                exact_two_sided_p=float(binomtest(gains, gains + losses, 0.5).pvalue) if gains + losses else 1.0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--output", type=Path, default=ROOT / "results/loss_diagnostics.json")
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261011)
    args = parser.parse_args()
    if args.bootstrap < 100:
        raise ValueError("At least 100 bootstrap replicates are required")
    paths = {rid: args.results / "classification" / rid / "predictions.csv" for rid in RUNS}
    loaded = {rid: read_predictions(path) for rid, path in paths.items()}
    ids, y = loaded[RUNS[0]][:2]
    for rid, (other_ids, other_y, _, _) in loaded.items():
        if other_ids != ids or not np.array_equal(other_y, y):
            raise ValueError(f"Paired image ordering/labels differ in {rid}")
    strata = [np.flatnonzero(y == k) for k in range(7)]
    if any(not len(s) for s in strata):
        raise ValueError("Every class needs test support")
    estimated, samples, cms = {}, {}, {}
    for rid, (_, _, pred, prob) in loaded.items():
        cm = confusion_batches(np.ones((1, len(y))), y, pred)
        cms[rid] = cm[0].astype(int)
        estimated[rid] = {k: float(v[0]) for k, v in scores(cm).items()}
        estimated[rid]["macro_auroc"] = float(auc_batches(np.ones((1, len(y))), y, prob)[0])
        samples[rid] = {k: np.empty(args.bootstrap) for k in estimated[rid]}
        saved = json.loads((args.results / "classification" / rid / "metrics.json").read_text())
        for key, value in estimated[rid].items():
            if key in saved and not np.isclose(value, saved[key], rtol=0, atol=1e-10):
                raise ValueError(f"Frozen {key} mismatch for {rid}")
    rng = np.random.default_rng(args.seed)
    for start in range(0, args.bootstrap, 250):
        stop = min(start + 250, args.bootstrap)
        weights = np.zeros((stop - start, len(y)))
        for indices in strata:
            weights[:, indices] = rng.multinomial(len(indices), np.full(len(indices), 1 / len(indices)), size=stop - start)
        for rid, (_, _, pred, prob) in loaded.items():
            current = scores(confusion_batches(weights, y, pred))
            current["macro_auroc"] = auc_batches(weights, y, prob)
            for key, value in current.items():
                samples[rid][key][start:stop] = value
    summaries = {rid: {k: dict(estimate=v, ci95=interval(samples[rid][k])) for k, v in values.items()}
                 for rid, values in estimated.items()}
    pairs = {}
    per_class_tests = []
    for rid in RUNS[1:]:
        other = loaded[rid][2]
        comparison = dict(delta_definition=f"{rid} minus cls_r50_ce", metrics={},
                          accuracy_mcnemar=mcnemar(y, loaded[RUNS[0]][2], other), per_class={})
        for key, value in estimated[rid].items():
            comparison["metrics"][key] = dict(delta=value - estimated[RUNS[0]][key],
                ci95=interval(samples[rid][key] - samples[RUNS[0]][key]))
        for k, name in enumerate(CLASSES):
            subset = y == k
            test = mcnemar(y[subset], loaded[RUNS[0]][2][subset], other[subset])
            test.update(support=int(subset.sum()),
                        correct_change=int(cms[rid][k, k] - cms[RUNS[0]][k, k]),
                        predicted_count_change=int(cms[rid][:, k].sum() - cms[RUNS[0]][:, k].sum()),
                        recall_delta=float(estimated[rid][f"recall_{name}"] - estimated[RUNS[0]][f"recall_{name}"]))
            comparison["per_class"][name] = test
            per_class_tests.append(test)
        differences = cms[rid] - cms[RUNS[0]]
        comparison["confusion_count_change"] = differences.tolist()
        changes = [dict(gt=CLASSES[k], pred=CLASSES[j], delta=int(differences[k, j]))
                   for k in range(7) for j in range(7) if j != k and differences[k, j]]
        comparison["largest_error_changes"] = sorted(changes, key=lambda x: -abs(x["delta"]))[:10]
        pairs[rid] = comparison
    for test, value in zip(per_class_tests, holm([r["exact_two_sided_p"] for r in per_class_tests])):
        test["holm_p_across_14_class_tests"] = value
    accuracy_tests = [p["accuracy_mcnemar"] for p in pairs.values()]
    for test, value in zip(accuracy_tests, holm([r["exact_two_sided_p"] for r in accuracy_tests])):
        test["holm_p_across_2_accuracy_tests"] = value
    run_audit = {}
    for rid in RUNS:
        metadata = json.loads((ROOT / "outputs" / rid / "metadata.json").read_text())
        history = json.loads((ROOT / "outputs" / rid / "history.json").read_text())
        run_audit[rid] = {k: metadata[k] for k in ["seed", "initialization", "best_epoch", "completed_epochs", "selection", "best_val_accuracy", "best_val_macro_f1", "best_checkpoint_sha256"]}
        run_audit[rid]["best_macro_f1_epoch_in_recorded_history"] = max(history, key=lambda x: x["val_macro_f1"])["epoch"]
        run_audit[rid]["lr_at_last_epoch"] = history[-1]["lr"]
    split = json.loads((ROOT / "splits/isic_split.json").read_text())
    counts = {part: [sum(row["label"] == k for row in rows) for k in range(7)] for part, rows in split.items() if isinstance(rows, list)}
    train_counts = np.array(counts["train"])
    inverse = train_counts.sum() / (7 * train_counts)
    # Descriptive batch-composition simulation; it is not a training change.
    batches = np.random.default_rng(args.seed + 1).multinomial(16, train_counts / train_counts.sum(), size=100000)
    contributions = batches * inverse
    result = dict(
        source="Frozen saved probabilities; no fitting, checkpoint changes, threshold changes or postprocessing",
        n=len(y), classes=CLASSES, seed=args.seed, bootstrap_replicates=args.bootstrap,
        uncertainty_protocol="Paired image bootstrap stratified by true class, observed support fixed; percentile 95% intervals. Identical resampling weights for every model.",
        interpretation_limits=["Intervals condition on these fixed fitted models and observed class composition; they do not capture training-seed uncertainty.",
            "Lesion/patient clustering is unavailable. Image-level intervals may be optimistic if images are correlated.",
            "Accuracy McNemar tests are exact two-sided; their two baseline comparisons use Holm correction. Fourteen exploratory per-class recall tests use a separate Holm family.",
            "Bootstrap intervals for multiple descriptive metrics are not multiplicity adjusted. Neither a small p-value nor a positive interval establishes clinical utility.",
            "Training and test prevalence differ. Changing a selection rule or loss based on these test diagnostics would require a new held-out test set for confirmatory evaluation."],
        input_sha256={str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path): digest(path) for path in paths.values()},
        manifest_sha256=digest(ROOT / "splits/evaluation_manifest.json"),
        summaries=summaries, baseline_comparisons=pairs, run_audit=run_audit,
        split_counts=counts, inverse_frequency_weights=inverse.tolist(),
        batch_weight_diagnostics=dict(batch_size=16, simulation_batches=100000,
            absent_probability=(1 - train_counts / train_counts.sum()) ** 16,
            expected_class_share_in_normalized_wce_batch=(contributions / contributions.sum(1, keepdims=True)).mean(0),
            note="PyTorch weighted mean CE divides each batch by its sum of target weights. Global weight rescaling cancels. Small-batch normalization does not generally yield exactly uniform expected class contribution."))
    result["batch_weight_diagnostics"]["absent_probability"] = result["batch_weight_diagnostics"]["absent_probability"].tolist()
    result["batch_weight_diagnostics"]["expected_class_share_in_normalized_wce_batch"] = result["batch_weight_diagnostics"]["expected_class_share_in_normalized_wce_batch"].tolist()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["model", "metric", "estimate", "ci95_low", "ci95_high", "delta_vs_ce", "delta_ci95_low", "delta_ci95_high"])
        writer.writeheader()
        for rid, values in summaries.items():
            for metric, value in values.items():
                delta = pairs.get(rid, {}).get("metrics", {}).get(metric)
                writer.writerow(dict(model=rid, metric=metric, estimate=value["estimate"], ci95_low=value["ci95"][0], ci95_high=value["ci95"][1],
                    delta_vs_ce=delta["delta"] if delta else "", delta_ci95_low=delta["ci95"][0] if delta else "", delta_ci95_high=delta["ci95"][1] if delta else ""))
    print(f"Verified {len(y)} paired frozen predictions; saved {args.output} and {csv_path}")
    for rid, pair in pairs.items():
        print(rid, {k: pair["metrics"][k] for k in ["accuracy", "macro_recall", "macro_f1"]}, pair["accuracy_mcnemar"])


if __name__ == "__main__":
    main()
