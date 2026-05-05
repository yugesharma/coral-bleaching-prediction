import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.metrics import (
    RocCurveDisplay,
    PrecisionRecallDisplay,
    confusion_matrix,
    ConfusionMatrixDisplay
)
import os


def plot_roc_curve(y_true, y_probs, model_name, output_dir):
    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(
        y_true, y_probs, name=model_name, ax=ax
    )
    ax.plot([0, 1], [0, 1], 'k--', label='Random')
    ax.set_title(f'ROC Curve — {model_name}')
    ax.legend()
    plt.tight_layout()
    path = os.path.join(output_dir, f'roc_{model_name}.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_pr_curve(y_true, y_probs, model_name, output_dir):
    fig, ax = plt.subplots(figsize=(6, 5))
    PrecisionRecallDisplay.from_predictions(
        y_true, y_probs, name=model_name, ax=ax
    )
    ax.set_title(f'PR Curve — {model_name}')
    ax.legend()
    plt.tight_layout()
    path = os.path.join(output_dir, f'pr_{model_name}.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_confusion_matrix(y_true, y_probs, threshold, model_name, output_dir):
    y_pred = (y_probs >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=['No Bleaching', 'Bleaching']
    )
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_title(f'Confusion Matrix — {model_name}')
    plt.tight_layout()
    path = os.path.join(output_dir, f'cm_{model_name}.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_training_history(history, model_name, output_dir):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(history['train_loss'], label='Train Loss')
    ax.plot(history['val_loss'],   label='Val Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title(f'Training History — {model_name}')
    ax.legend()
    plt.tight_layout()
    path = os.path.join(output_dir, f'history_{model_name}.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_attention_weights(attn_weights, depth_levels, model_name, output_dir):

    mean_weights = attn_weights.mean(axis=0)  

    fig, ax = plt.subplots(figsize=(5, 7))
    ax.plot(mean_weights, depth_levels)
    ax.invert_yaxis()                         
    ax.set_xlabel('Mean Attention Weight')
    ax.set_ylabel('Depth (m)')
    ax.set_title(f'Attention Weights — {model_name}')
    ax.axhline(y=30,  color='r', linestyle='--', alpha=0.5, label='30m')
    ax.axhline(y=100, color='b', linestyle='--', alpha=0.5, label='100m')
    ax.legend()
    plt.tight_layout()
    path = os.path.join(output_dir, f'attention_{model_name}.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_all_roc_curves(models_data, output_dir):
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, (y_true, y_probs) in models_data.items():
        RocCurveDisplay.from_predictions(
            y_true, y_probs, name=name, ax=ax
        )
    ax.plot([0, 1], [0, 1], 'k--', label='Random')
    ax.set_title('ROC Curves — All Models')
    ax.legend()
    plt.tight_layout()
    path = os.path.join(output_dir, 'roc_all_models.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_all_pr_curves(models_data, output_dir):
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, (y_true, y_probs) in models_data.items():
        PrecisionRecallDisplay.from_predictions(
            y_true, y_probs, name=name, ax=ax
        )
    ax.set_title('PR Curves — All Models')
    ax.legend()
    plt.tight_layout()
    path = os.path.join(output_dir, 'pr_all_models.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")