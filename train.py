"""Cross-validate several classic ML models on the battle features and write a submission.

Base models use the hyperparameters in best_params.json when it exists (see tune.py).
Each team's past win rate is added inside every fold, out-of-fold, so no battle sees its own label.
The base models' out-of-fold probabilities are combined two ways: a plain average and a stacked
logistic regression; the submission uses whichever scores better in cross-validation.

Usage: python train.py
"""
import json
import os
import shutil
import warnings
from collections import Counter

import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.base import clone
from sklearn.ensemble import (ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier,
                              VotingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features import load_jsonl, make_features

SEED = 42
N_FOLDS = 5
PARAMS_FILE = "best_params.json"
# XGBoost is averaged over this many random seeds to reduce the variance of a single run.
XGB_SEEDS = 5
# Models left out of the final ensemble (random forest only cancelled out extra trees in the stack).
EXCLUDE = {"random_forest"}
TEAM_PRIOR = 20
TEAM_COLS = ["p1_team_winrate", "p1_team_count", "p2_team_winrate", "p2_team_count"]

# Our data lives on the CPU, so a CUDA booster predicts through a DMatrix copy; harmless here.
warnings.filterwarnings("ignore", message=".*Falling back to prediction using DMatrix.*")


def xgb_device():
    """'cuda' when xgboost has CUDA support and an NVIDIA GPU is visible, else 'cpu'.

    Override with the XGB_DEVICE environment variable. Only XGBoost can use the GPU;
    the scikit-learn models always run on the CPU.
    """
    if os.environ.get("XGB_DEVICE"):
        return os.environ["XGB_DEVICE"]
    import xgboost
    has_cuda = xgboost.build_info().get("USE_CUDA", False)
    return "cuda" if has_cuda and shutil.which("nvidia-smi") else "cpu"


def get_models(tuned=True, device=None, xgb_seeds=XGB_SEEDS, exclude=()):
    best = {}
    if tuned and os.path.exists(PARAMS_FILE):
        with open(PARAMS_FILE) as f:
            best = json.load(f)
        print(f"Using tuned hyperparameters from {PARAMS_FILE} for: {', '.join(best)}")

    models = {
        "logreg": make_pipeline(
            StandardScaler(), LogisticRegression(C=0.1, solver="saga", max_iter=5000)),
        "random_forest": RandomForestClassifier(
            n_estimators=800, min_samples_leaf=3, max_features=0.3, n_jobs=-1, random_state=SEED),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=800, min_samples_leaf=3, max_features=0.4, n_jobs=-1, random_state=SEED),
        "hist_gbm": HistGradientBoostingClassifier(
            learning_rate=0.03, max_iter=600, max_leaf_nodes=15, min_samples_leaf=40,
            l2_regularization=1.0, random_state=SEED),
    }
    for name, model in models.items():
        params = dict(best.get(name, {}))
        # Forests were tuned with 500 trees for speed; more trees only reduce variance.
        if "n_estimators" in params and name in ("random_forest", "extra_trees"):
            params["n_estimators"] = 1000
        model.set_params(**params)

    try:
        from xgboost import XGBClassifier
        device = device or xgb_device()
        xgb = XGBClassifier(
            n_estimators=800, learning_rate=0.02, max_depth=4, subsample=0.8, colsample_bytree=0.6,
            min_child_weight=5, reg_lambda=2.0, n_jobs=-1, random_state=SEED, eval_metric="logloss",
            device=device).set_params(**best.get("xgboost", {}))
        if xgb_seeds > 1:
            xgb = VotingClassifier(
                [(f"seed{s}", clone(xgb).set_params(random_state=SEED + s)) for s in range(xgb_seeds)],
                voting="soft")
        models["xgboost"] = xgb
        print(f"xgboost device: {device}, seeds: {xgb_seeds}")
    except ImportError:
        print("xgboost not installed, skipping it (pip install xgboost to enable)")
    return {name: m for name, m in models.items() if name not in exclude}


def team_rates(keys_fit, y_fit, keys_apply):
    """Smoothed win rate (pulled toward 0.5) and battle count of each battle's p1 and p2 team."""
    out = np.zeros((len(keys_apply), 4))
    for j in range(2):
        wins, n = Counter(), Counter()
        for k, label in zip(keys_fit, y_fit):
            if k[j] is not None:
                wins[k[j]] += label
                n[k[j]] += 1
        for i, k in enumerate(keys_apply):
            c = n[k[j]] if k[j] is not None else 0
            out[i, 2 * j] = (wins[k[j]] + 0.5 * TEAM_PRIOR) / (c + TEAM_PRIOR) if c else 0.5
            out[i, 2 * j + 1] = c
    return out


def add_team_rates(X_fit, X_apply, keys_fit, keys_apply, y_fit):
    """Append team win rates. Rows used for fitting get out-of-fold rates so no battle sees its own label."""
    enc_fit = np.zeros((len(keys_fit), len(TEAM_COLS)))
    inner = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED + 2)
    for a, b in inner.split(X_fit, y_fit):
        enc_fit[b] = team_rates([keys_fit[i] for i in a], y_fit[a], [keys_fit[i] for i in b])
    enc_fit[:, [1, 3]] *= N_FOLDS / (N_FOLDS - 1)  # counts came from 4/5 of the rows
    enc_apply = team_rates(keys_fit, y_fit, keys_apply)
    return (X_fit.assign(**dict(zip(TEAM_COLS, enc_fit.T))),
            X_apply.assign(**dict(zip(TEAM_COLS, enc_apply.T))))


def stack_features(probs):
    return logit(np.clip(probs, 1e-4, 1 - 1e-4))


def stack_oof(oof, y):
    """Out-of-fold predictions of the stacking logistic regression (second CV, different seed)."""
    return cross_val_predict(
        LogisticRegression(C=1.0), stack_features(oof), y, method="predict_proba",
        cv=StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED + 1))[:, 1]


def report(name, y, p):
    print(f"{name:22s} {accuracy_score(y, p > 0.5):7.4f} {roc_auc_score(y, p):7.4f} {log_loss(y, p):8.4f}")


def main():
    train = load_jsonl("train.jsonl")
    test = load_jsonl("test.jsonl")
    X, X_test, keys, keys_test = make_features(train, test, "type_chart_gen2.csv", return_keys=True)
    y = np.array([b["player_won"] for b in train])
    print(f"{X.shape[1] + len(TEAM_COLS)} features, {len(X)} train battles, {len(X_test)} test battles\n")

    # Evaluate every model (including excluded ones) so the ablation below can justify EXCLUDE.
    models = get_models()
    names = list(models)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros((len(X), len(models)))
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        X_tr, X_va = add_team_rates(X.iloc[tr_idx], X.iloc[va_idx],
                                    [keys[i] for i in tr_idx], [keys[i] for i in va_idx], y[tr_idx])
        for j, model in enumerate(models.values()):
            m = clone(model).fit(X_tr, y[tr_idx])
            oof[va_idx, j] = m.predict_proba(X_va)[:, 1]
        print(f"fold {fold + 1}/{N_FOLDS} done")

    print(f"\n{'model':22s} {'acc':>7s} {'auc':>7s} {'logloss':>8s}")
    for j, name in enumerate(names):
        report(name, y, oof[:, j])
    used = [j for j, n in enumerate(names) if n not in EXCLUDE]
    mean_oof = oof[:, used].mean(axis=1)
    report("ensemble(mean)", y, mean_oof)
    stacked = stack_oof(oof[:, used], y)
    report("stacked(logreg)", y, stacked)
    print("\nStacking ablation (all models, then dropping one at a time):")
    report("  all models", y, stack_oof(oof, y))
    for j, name in enumerate(names):
        report(f"  without {name}", y, stack_oof(np.delete(oof, j, axis=1), y))

    meta = LogisticRegression(C=1.0).fit(stack_features(oof[:, used]), y)
    print("\nstack weights: " + ", ".join(f"{names[j]}={w:.2f}" for j, w in zip(used, meta.coef_[0])))

    # Fit the used models on all training data, then combine with the better-scoring method.
    X_full, X_test_full = add_team_rates(X, X_test, keys, keys_test, y)
    final = [m for n, m in models.items() if n not in EXCLUDE]
    test_probs = np.column_stack([clone(m).fit(X_full, y).predict_proba(X_test_full)[:, 1] for m in final])
    use_stack = accuracy_score(y, stacked > 0.5) > accuracy_score(y, mean_oof > 0.5)
    if use_stack:
        test_pred = meta.predict_proba(stack_features(test_probs))[:, 1]
    else:
        test_pred = test_probs.mean(axis=1)
    sub = pd.DataFrame({"battle_id": [b["battle_id"] for b in test],
                        "player_won": (test_pred > 0.5).astype(int)})
    sub.to_csv("submission.csv", index=False)
    print(f"\nWrote submission.csv using {'stacking' if use_stack else 'mean'} of "
          f"{', '.join(n for n in names if n not in EXCLUDE)} "
          f"({len(sub)} rows, predicted win rate {sub.player_won.mean():.3f})")


if __name__ == "__main__":
    main()
