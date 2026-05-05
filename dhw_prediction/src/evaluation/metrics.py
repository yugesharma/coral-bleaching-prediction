import json
import os
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    confusion_matrix
)


def find_best_threshold(y_true, y_probs):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_probs)
    f1_scores = (2 * precisions * recalls /
                 (precisions + recalls + 1e-8))
    best_idx   = f1_scores[:-1].argmax()
    return thresholds[best_idx]


def compute_metrics(y_true, y_probs, threshold):
    y_pred = (y_probs >= threshold).astype(int)

    roc_auc = roc_auc_score(y_true, y_probs)
    pr_auc  = average_precision_score(y_true, y_probs)
    cm      = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)

    metrics = {
        'roc_auc':   round(roc_auc, 4),
        'pr_auc':    round(pr_auc, 4),
        'threshold': round(float(threshold), 4),
        'accuracy':  round(accuracy, 4),
        'precision': round(precision, 4),
        'recall':    round(recall, 4),
        'f1':        round(f1, 4),
        'tp': int(tp), 'tn': int(tn),
        'fp': int(fp), 'fn': int(fn)
    }

    return metrics


def evaluate(y_true_val, y_probs_val, y_true_test, y_probs_test):
    threshold = find_best_threshold(y_true_val, y_probs_val)
    print(f"Best threshold (val): {threshold:.4f}")

    val_metrics  = compute_metrics(y_true_val,  y_probs_val,  threshold)
    test_metrics = compute_metrics(y_true_test, y_probs_test, threshold)

    print("\n--- VAL METRICS ---")
    _print_metrics(val_metrics)

    print("\n--- TEST METRICS ---")
    _print_metrics(test_metrics)

    return val_metrics, test_metrics


def _print_metrics(m):
    print(f"  ROC-AUC:   {m['roc_auc']}")
    print(f"  PR-AUC:    {m['pr_auc']}")
    print(f"  F1:        {m['f1']}")
    print(f"  Precision: {m['precision']}")
    print(f"  Recall:    {m['recall']}")
    print(f"  Accuracy:  {m['accuracy']}")
    print(f"  TP:{m['tp']}  TN:{m['tn']}  FP:{m['fp']}  FN:{m['fn']}")


def save_metrics(val_metrics, test_metrics, output_dir):
    """Save val and test metrics to a single json file."""
    os.makedirs(output_dir, exist_ok=True)
    output = {'val': val_metrics, 'test': test_metrics}
    path = os.path.join(output_dir, 'metrics.json')
    with open(path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nMetrics saved to {path}")