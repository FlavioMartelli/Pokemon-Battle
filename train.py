"""
Training of the models and creation of the final csv to submit
The models are trained using the parameters contained in best_params.json if it exists.
Each team's past win rate is added inside every fold, out-of-fold, so no battle sees its own label.
The models' out-of-fold probabilities are combined two ways: a plain average and a stacked
logistic regression and the submission uses whichever scores better in cross-validation.

Usage: python train.py (preferrably after tune.py)
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

# Initial setup
SEED = 42
N_FOLDS = 5
PARAMS_FILE = "best_params.json"
XGB_SEEDS = 5 # XGB averaged over 5 to reduce variance
EXCLUDE = {"random_forest"} # Excluded because it was very similar to extra_trees but worse, it is checked at the end
TEAM_PRIOR = 20 # Bayesian prior for team win rates, to prevent overfitting due to the small sample size
TEAM_COLS = ["p1_team_winrate", "p1_team_count", "p2_team_winrate", "p2_team_count"]

# Tired of seeing this warning, being harmless, so i have suppressed it
warnings.filterwarnings("ignore", message=".*Falling back to prediction using DMatrix.*")

# Kinda forcing XGB to run on GPU if possible
def xgb_device():
    if os.environ.get("XGB_DEVICE"):
        return os.environ["XGB_DEVICE"]
    import xgboost
    has_cuda = xgboost.build_info().get("USE_CUDA", False)
    return "cuda" if has_cuda and shutil.which("nvidia-smi") else "cpu"


def get_models(tuned=True, device=None, xgb_seeds=XGB_SEEDS, exclude=()):
    best = {}

    # If tuning was done, read the best_params json file
    if tuned and os.path.exists(PARAMS_FILE):
        with open(PARAMS_FILE) as f:
            best = json.load(f)
        print(f"Using tuned hyperparameters from {PARAMS_FILE} for: {', '.join(best)}")

    # Initizlize the models
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

    # Setting up the models with their best parameters
    for name, model in models.items():
        params = dict(best.get(name, {}))
        if "n_estimators" in params and name in ("random_forest", "extra_trees"):
            params["n_estimators"] = 1000 # Set lower to make the training faster, a higher value will only decrease variance
        model.set_params(**params)

    try:

        # Configuration of XGB with best parameters
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

    # Return all the models, except for the excluded one
    return {name: m for name, m in models.items() if name not in exclude}

# Computation of the Bayesina smoothed win rate, outputs p1 and p2 win rates and counts in this order for both
def team_rates(keys_fit, y_fit, keys_apply):
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

# Adds the team win rates to the features. Each fold only recieves the rates obtained from other folds (out of fold) to avoid any leakage,
# also the counts are scaled accounting for the fact that each fold only sees 80% of the data
def add_team_rates(X_fit, X_apply, keys_fit, keys_apply, y_fit):
    enc_fit = np.zeros((len(keys_fit), len(TEAM_COLS)))
    inner = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED + 2)
    for a, b in inner.split(X_fit, y_fit):
        enc_fit[b] = team_rates([keys_fit[i] for i in a], y_fit[a], [keys_fit[i] for i in b])
    enc_fit[:, [1, 3]] *= N_FOLDS / (N_FOLDS - 1)  # counts came from 4/5 of the rows
    enc_apply = team_rates(keys_fit, y_fit, keys_apply)
    return (X_fit.assign(**dict(zip(TEAM_COLS, enc_fit.T))),
            X_apply.assign(**dict(zip(TEAM_COLS, enc_apply.T))))

# Converts probabilities to logit space
def stack_features(probs):
    return logit(np.clip(probs, 1e-4, 1 - 1e-4))

# Stacks the out of folds probabolities
def stack_oof(oof, y):
    return cross_val_predict(
        LogisticRegression(C=1.0), stack_features(oof), y, method="predict_proba",
        cv=StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED + 1))[:, 1]

# Reports the accuracy, AUC and nll of any model, giving predicted probabilities and true lables
def report(name, y, p):
    print(f"{name:22s} {accuracy_score(y, p > 0.5):7.4f} {roc_auc_score(y, p):7.4f} {log_loss(y, p):8.4f}")


def main():

    # Loading and extracting area
    train = load_jsonl("train.jsonl")
    test = load_jsonl("test.jsonl")
    X, X_test, keys, keys_test = make_features(train, test, "type_chart_gen2.csv", return_keys=True)
    y = np.array([b["player_won"] for b in train])
    print(f"{X.shape[1] + len(TEAM_COLS)} features, {len(X)} train battles, {len(X_test)} test battles\n")

    # Evaluate every model, including the random forest to justify its exclusion.
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

    # Reports the results of each model, mean of the used models and stacked logistic regression of the models used
    print(f"\n{'model':22s} {'acc':>7s} {'auc':>7s} {'logloss':>8s}")
    for j, name in enumerate(names):
        report(name, y, oof[:, j])
    used = [j for j, n in enumerate(names) if n not in EXCLUDE]
    mean_oof = oof[:, used].mean(axis=1)
    report("ensemble(mean)", y, mean_oof)
    stacked = stack_oof(oof[:, used], y)
    report("stacked(logreg)", y, stacked)

    # To justify the exlusion of random forest, here we do an ablation study on the stacking. We remove one model at the time 
    # and check the performance, seeing in the end that random forest is the one that can get rid of
    print("\nStacking ablation (all models, then dropping one at a time):")
    report("  all models", y, stack_oof(oof, y))
    for j, name in enumerate(names):
        report(f"  without {name}", y, stack_oof(np.delete(oof, j, axis=1), y))

    # Training of the final model used to create the final csv
    meta = LogisticRegression(C=1.0).fit(stack_features(oof[:, used]), y)
    print("\nstack weights: " + ", ".join(f"{names[j]}={w:.2f}" for j, w in zip(used, meta.coef_[0])))

    # Fitting of the final model on the whole training set
    X_full, X_test_full = add_team_rates(X, X_test, keys, keys_test, y)
    final = [m for n, m in models.items() if n not in EXCLUDE]
    test_probs = np.column_stack([clone(m).fit(X_full, y).predict_proba(X_test_full)[:, 1] for m in final])

    # Checks whether the stacking or averaging is better and decides which one to use and creation of the final csv
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
