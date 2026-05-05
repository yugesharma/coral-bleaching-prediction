import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

RUNS_DIR = 'experiments/runs'
MODELS   = ['xgboost', 'dnn', 'cnn_attention']
OUTPUT   = 'results'
os.makedirs(OUTPUT, exist_ok=True)


def load_metrics(model_name):
    path = os.path.join(RUNS_DIR, model_name, 'metrics.json')
    with open(path) as f:
        return json.load(f)


def build_comparison_table():
    rows = []

    for model in MODELS:
        try:
            m = load_metrics(model)
            rows.append({
                'model':        model,
                'val_roc_auc':  m['val']['roc_auc'],
                'val_pr_auc':   m['val']['pr_auc'],
                'val_f1':       m['val']['f1'],
                'test_roc_auc': m['test']['roc_auc'],
                'test_pr_auc':  m['test']['pr_auc'],
                'test_f1':      m['test']['f1'],
                'test_precision': m['test']['precision'],
                'test_recall':    m['test']['recall'],
                'threshold':      m['val']['threshold'],
            })
        except FileNotFoundError:
            print(f"Warning: metrics not found for {model} — skipping")

    df = pd.DataFrame(rows).set_index('model')
    return df


def plot_comparison_bar(df):
    metrics  = ['test_roc_auc', 'test_pr_auc', 'test_f1']
    labels   = ['ROC-AUC', 'PR-AUC', 'F1']
    x        = np.arange(len(metrics))
    width    = 0.25
    fig, ax  = plt.subplots(figsize=(9, 5))

    for i, (model, row) in enumerate(df.iterrows()):
        values = [row[m] for m in metrics]
        ax.bar(x + i * width, values, width, label=model)

    ax.set_xticks(x + width)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel('Score')
    ax.set_title('Model Comparison — Test Set')
    ax.legend()
    plt.tight_layout()

    path = os.path.join(OUTPUT, 'comparison_bar.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_comparison_roc(df):
    """
    Load saved probabilities per model and plot ROC curves together.
    Requires probs to be saved during experiments — see note below.
    """

    pass


def main():
    print("Loading metrics from all runs...\n")
    df = build_comparison_table()

    print("="*65)
    print(df.to_string())
    print("="*65)

    # Save table as CSV
    csv_path = os.path.join(OUTPUT, 'comparison.csv')
    df.to_csv(csv_path)
    print(f"\nTable saved: {csv_path}")

    # Bar chart
    plot_comparison_bar(df)


if __name__ == '__main__':
    main()