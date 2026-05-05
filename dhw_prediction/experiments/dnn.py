import sys
import os
sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../')
))

import json
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from src.data.dataset import TabularDataset
from src.data.preprocessing import fit_scalers, apply_scalers, save_scalers
from src.models import MLP
from src.training.trainer import train, save_history
from src.training.hpo import run_hpo
from src.evaluation.metrics import evaluate, save_metrics
from src.evaluation.plots import (
    plot_roc_curve, plot_pr_curve,
    plot_confusion_matrix, plot_training_history
)

RESULTS_DIR = 'experiments/runs/dnn'
os.makedirs(RESULTS_DIR, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_probs(model, loader, device):
    """Get predicted probabilities from MLP."""
    model.eval()
    all_probs  = []
    all_labels = []

    with torch.no_grad():
        for x_tab, y in loader:
            x_tab = x_tab.to(device)
            logits = model(x_tab)
            probs  = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(y.numpy())

    return np.concatenate(all_probs), np.concatenate(all_labels)


def run():
    print(f"Device: {DEVICE}")

    train_df = pd.read_pickle('data/processed/train.pkl')
    val_df   = pd.read_pickle('data/processed/val.pkl')
    test_df  = pd.read_pickle('data/processed/test.pkl')

    tab_scaler, seq_scalers = fit_scalers(train_df)
    train_df = apply_scalers(train_df, tab_scaler, seq_scalers)
    val_df   = apply_scalers(val_df,   tab_scaler, seq_scalers)
    test_df  = apply_scalers(test_df,  tab_scaler, seq_scalers)

    save_scalers(tab_scaler, seq_scalers, RESULTS_DIR)

    print("\nRunning DNN HPO")
    best_params = run_hpo(
        model_type='mlp',
        train_df=train_df,
        val_df=val_df,
        device=DEVICE,
        output_dir=RESULTS_DIR,
        n_trials=50
    )

    with open(os.path.join(RESULTS_DIR, 'best_params.json'), 'w') as f:
        json.dump(best_params, f, indent=2)

    # ── Retrain with best params
    print("\nRetraining with best params...")
    hidden_dims = [best_params['hidden_dim']] * best_params['n_layers']

    train_ds     = TabularDataset(train_df)
    val_ds       = TabularDataset(val_df)
    test_ds      = TabularDataset(test_df)
    train_loader = DataLoader(
        train_ds, batch_size=best_params['batch_size'], shuffle=True
    )
    val_loader   = DataLoader(val_ds,  batch_size=64, shuffle=False)
    test_loader  = DataLoader(test_ds, batch_size=64, shuffle=False)

    model = MLP(
        input_dim=12,
        hidden_dims=hidden_dims,
        dropout=best_params['dropout']
    ).to(DEVICE)

    n_neg      = (train_df['label'] == 0).sum()
    n_pos      = (train_df['label'] == 1).sum()
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32).to(DEVICE)
    loss_fn    = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer  = torch.optim.Adam(model.parameters(), lr=best_params['lr'])

    model, history = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        n_epochs=100,
        patience=15,
        device=DEVICE,
        model_type='mlp',
        save_path=os.path.join(RESULTS_DIR, 'best_model.pt')
    )

    save_history(history, RESULTS_DIR)

    y_val_probs,  y_val  = get_probs(model, val_loader,  DEVICE)
    y_test_probs, y_test = get_probs(model, test_loader, DEVICE)

    val_metrics, test_metrics = evaluate(
        y_val, y_val_probs, y_test, y_test_probs
    )
    save_metrics(val_metrics, test_metrics, RESULTS_DIR)

    # ── Plots
    figures_dir = os.path.join(RESULTS_DIR, 'figures')
    os.makedirs(figures_dir, exist_ok=True)

    plot_roc_curve(y_test, y_test_probs, 'DNN', figures_dir)
    plot_pr_curve(y_test, y_test_probs, 'DNN', figures_dir)
    plot_confusion_matrix(
        y_test, y_test_probs,
        val_metrics['threshold'], 'DNN', figures_dir
    )
    plot_training_history(history, 'DNN', figures_dir)

    np.save(os.path.join(RESULTS_DIR, 'y_test_probs.npy'), y_test_probs)
    np.save(os.path.join(RESULTS_DIR, 'y_test.npy'),y_test)

if __name__ == '__main__':
    run()