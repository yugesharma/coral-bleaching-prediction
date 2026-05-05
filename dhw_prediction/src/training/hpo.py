import optuna
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score

import sys
import os
sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../../')
))

from src.data.dataset import TabularDataset, ProfileDataset
from src.models import MLP, CNN1DWithAttention
from src.training.trainer import train


def get_pos_weight(y_train):
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    return torch.tensor([n_neg / n_pos], dtype=torch.float32)

def mlp_objective(trial, train_df, val_df, device, output_dir):
    n_layers    = trial.suggest_int('n_layers', 1, 3)
    hidden_dim  = trial.suggest_categorical('hidden_dim', [32, 64, 128])
    dropout     = trial.suggest_float('dropout', 0.1, 0.5)
    lr          = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    batch_size  = trial.suggest_categorical('batch_size', [32, 64])

    hidden_dims = [hidden_dim] * n_layers

    train_ds     = TabularDataset(train_df)
    val_ds       = TabularDataset(val_df)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    model = MLP(
        input_dim=12,
        hidden_dims=hidden_dims,
        dropout=dropout
    ).to(device)

    pos_weight = get_pos_weight(train_df['label']).to(device)
    loss_fn    = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer  = torch.optim.Adam(model.parameters(), lr=lr)

    save_path = os.path.join(output_dir, f'trial_{trial.number}.pt')

    model, _ = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        n_epochs=50,           
        patience=10,
        device=device,
        model_type='mlp',
        save_path=save_path
    )

    model.eval()
    all_probs  = []
    all_labels = []

    with torch.no_grad():
        for x_tab, y in val_loader:
            x_tab = x_tab.to(device)
            probs = model(x_tab).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(y.numpy())

    y_probs  = np.concatenate(all_probs)
    y_true   = np.concatenate(all_labels)
    pr_auc   = average_precision_score(y_true, y_probs)

    if os.path.exists(save_path):
        os.remove(save_path)

    return pr_auc

def cnn_objective(trial, train_df, val_df, device, output_dir):
    n_filters           = trial.suggest_categorical('n_filters', [16, 32, 64])
    kernel_size         = trial.suggest_categorical('kernel_size', [3, 5, 7])
    n_conv_layers       = trial.suggest_int('n_conv_layers', 1, 3)
    attention_hidden    = trial.suggest_categorical('attention_hidden_dim', [8, 16, 32])
    dropout             = trial.suggest_float('dropout', 0.1, 0.5)
    lr                  = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    batch_size          = trial.suggest_categorical('batch_size', [16, 32])

    train_ds     = ProfileDataset(train_df)
    val_ds       = ProfileDataset(val_df)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    model = CNN1DWithAttention(
        n_depths=150,
        n_channels=2,
        n_filters=n_filters,
        kernel_size=kernel_size,
        n_conv_layers=n_conv_layers,
        attention_hidden_dim=attention_hidden,
        tab_input_dim=12,
        dropout=dropout
    ).to(device)

    pos_weight = get_pos_weight(train_df['label']).to(device)
    loss_fn    = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer  = torch.optim.Adam(model.parameters(), lr=lr)

    save_path = os.path.join(output_dir, f'trial_{trial.number}.pt')

    model, _ = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        n_epochs=50,
        patience=10,
        device=device,
        model_type='cnn',
        save_path=save_path
    )

    model.eval()
    all_probs  = []
    all_labels = []

    with torch.no_grad():
        for x_seq, x_tab, y in val_loader:
            x_seq = x_seq.to(device)
            x_tab = x_tab.to(device)
            probs, _ = model(x_seq, x_tab)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(y.numpy())

    y_probs = np.concatenate(all_probs)
    y_true  = np.concatenate(all_labels)
    pr_auc  = average_precision_score(y_true, y_probs)

    if os.path.exists(save_path):
        os.remove(save_path)

    return pr_auc

def run_hpo(model_type, train_df, val_df, device, output_dir, n_trials=50):

    os.makedirs(output_dir, exist_ok=True)

    if model_type == 'mlp':
        objective = lambda trial: mlp_objective(
            trial, train_df, val_df, device, output_dir
        )
    elif model_type == 'cnn':
        objective = lambda trial: cnn_objective(
            trial, train_df, val_df, device, output_dir
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"\nBest PR-AUC: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")

    return study.best_params