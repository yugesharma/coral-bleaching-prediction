import sys
import os
sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../')
))

import json
import numpy as np
import pandas as pd
import optuna
from xgboost import XGBClassifier
from sklearn.metrics import average_precision_score

from src.data.preprocessing import fit_scalers, apply_scalers, TABULAR_FEATURES
from src.evaluation.metrics import evaluate, save_metrics
from src.evaluation.plots import (
    plot_roc_curve, plot_pr_curve, plot_confusion_matrix
)

RESULTS_DIR = 'experiments/runs/xgboost-onlydhw'
os.makedirs(RESULTS_DIR, exist_ok=True)


def get_pos_weight(y_train):
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    return n_neg / n_pos


def xgb_objective(trial, X_train, y_train, X_val, y_val):
    params = {
        'n_estimators':      trial.suggest_int('n_estimators', 100, 500),
        'max_depth':         trial.suggest_int('max_depth', 3, 6),
        'learning_rate':     trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'subsample':         trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree':  trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'scale_pos_weight':  get_pos_weight(y_train),
        'eval_metric':       'aucpr',
        'early_stopping_rounds': 20,
        'random_state':      42
    }

    model = XGBClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    y_probs = model.predict_proba(X_val)[:, 1]
    return average_precision_score(y_val, y_probs)


def run():
    train_df = pd.read_pickle('data/processed/train.pkl')
    val_df   = pd.read_pickle('data/processed/val.pkl')
    test_df  = pd.read_pickle('data/processed/test.pkl')

    tab_scaler, seq_scalers = fit_scalers(train_df)
    train_df = apply_scalers(train_df, tab_scaler, seq_scalers)
    val_df   = apply_scalers(val_df,   tab_scaler, seq_scalers)
    test_df  = apply_scalers(test_df,  tab_scaler, seq_scalers)

    X_train = train_df[TABULAR_FEATURES].values
    y_train = train_df['label'].values
    X_val   = val_df[TABULAR_FEATURES].values
    y_val   = val_df['label'].values
    X_test  = test_df[TABULAR_FEATURES].values
    y_test  = test_df['label'].values

    print("Running XGBoost HPO")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='maximize')
    study.optimize(
        lambda trial: xgb_objective(trial, X_train, y_train, X_val, y_val),
        n_trials=50,
        show_progress_bar=True
    )

    best_params = study.best_params
    print(f"\nBest PR-AUC (val): {study.best_value:.4f}")
    print(f"Best params: {best_params}")

    with open(os.path.join(RESULTS_DIR, 'best_params.json'), 'w') as f:
        json.dump(best_params, f, indent=2)

    print("\nRetraining with best params")
    best_params['scale_pos_weight'] = get_pos_weight(y_train)
    best_params['eval_metric']      = 'aucpr'
    best_params['early_stopping_rounds'] = 20
    best_params['random_state']     = 42

    model = XGBClassifier(**best_params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    model.save_model(os.path.join(RESULTS_DIR, 'best_model.ubj'))

    y_val_probs  = model.predict_proba(X_val)[:, 1]
    y_test_probs = model.predict_proba(X_test)[:, 1]

    val_metrics, test_metrics = evaluate(
        y_val, y_val_probs, y_test, y_test_probs
    )
    save_metrics(val_metrics, test_metrics, RESULTS_DIR)

    # ── Plots
    figures_dir = os.path.join(RESULTS_DIR, 'figures')
    os.makedirs(figures_dir, exist_ok=True)

    plot_roc_curve(y_test, y_test_probs, 'XGBoost', figures_dir)
    plot_pr_curve(y_test, y_test_probs, 'XGBoost', figures_dir)
    plot_confusion_matrix(
        y_test, y_test_probs,
        val_metrics['threshold'], 'XGBoost', figures_dir
    )

    fi = pd.Series(
        model.feature_importances_,
        index=TABULAR_FEATURES
    ).sort_values(ascending=False)
    print("\nFeature Importance:")
    print(fi.round(3))

    fi.to_json(os.path.join(RESULTS_DIR, 'feature_importance.json'))

    np.save(os.path.join(RESULTS_DIR, 'y_test_probs.npy'), y_test_probs)
    np.save(os.path.join(RESULTS_DIR, 'y_test.npy'),y_test)

if __name__ == '__main__':
    run()