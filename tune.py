"""Randomized hyperparameter search for each base model; writes best_params.json for train.py.

Usage: python tune.py [n_iter]

Tuning uses a different CV split (seed 7) than the evaluation in train.py (seed 42),
so train.py's scores are less optimistic than the search's own best scores.
"""
import json
import sys
import time

import numpy as np
from scipy.stats import loguniform, randint, uniform
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from features import load_jsonl, make_features
from train import get_models

TUNE_SEED = 7

SPACES = {
    "logreg": {
        "logisticregression__C": loguniform(1e-3, 3),
        "logisticregression__l1_ratio": uniform(0, 1),
    },
    "random_forest": {
        "n_estimators": [500],
        "max_features": uniform(0.05, 0.45),
        "min_samples_leaf": randint(1, 25),
        "max_depth": [None, 8, 12, 16, 24],
    },
    "extra_trees": {
        "n_estimators": [500],
        "max_features": uniform(0.05, 0.6),
        "min_samples_leaf": randint(1, 25),
        "max_depth": [None, 8, 12, 16, 24],
    },
    "hist_gbm": {
        "learning_rate": loguniform(0.005, 0.1),
        "max_iter": randint(200, 1500),
        "max_leaf_nodes": randint(4, 48),
        "min_samples_leaf": randint(20, 300),
        "l2_regularization": loguniform(1e-2, 30),
        "max_features": uniform(0.2, 0.8),
    },
    "xgboost": {
        "n_estimators": randint(200, 1500),
        "learning_rate": loguniform(0.005, 0.1),
        "max_depth": randint(2, 7),
        "min_child_weight": loguniform(1, 50),
        "subsample": uniform(0.5, 0.5),
        "colsample_bytree": uniform(0.2, 0.8),
        "reg_lambda": loguniform(0.1, 30),
    },
}


def to_json(v):
    return v.item() if isinstance(v, np.generic) else v


def main():
    n_iter = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    train = load_jsonl("train.jsonl")
    test = load_jsonl("test.jsonl")
    X, _ = make_features(train, test, "type_chart_gen2.csv")
    y = np.array([b["player_won"] for b in train])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=TUNE_SEED)

    # XGBoost is tuned on the CPU: on this small dataset 20 parallel CPU fits beat one GPU,
    # which runs fits one at a time (benchmarked: 41s vs 57s for 20 fits on an RTX 3060).
    models = get_models(tuned=False, device="cpu", xgb_seeds=1)
    for name in ("random_forest", "extra_trees", "xgboost"):
        if name in models:  # parallelise over search candidates instead of inside each model
            models[name].set_params(n_jobs=1)

    best = {}
    for name, model in models.items():
        t0 = time.time()
        search = RandomizedSearchCV(
            model, SPACES[name], n_iter=n_iter, cv=cv, n_jobs=-1, random_state=TUNE_SEED,
            scoring={"logloss": "neg_log_loss", "acc": "accuracy"}, refit="logloss")
        search.fit(X, y)
        i = search.best_index_
        acc = search.cv_results_["mean_test_acc"][i]
        best[name] = {k: to_json(v) for k, v in search.best_params_.items()}
        print(f"{name:15s} logloss {-search.best_score_:.4f} acc {acc:.4f} "
              f"({time.time() - t0:.0f}s) {best[name]}", flush=True)

    with open("best_params.json", "w") as f:
        json.dump(best, f, indent=2)
    print("Wrote best_params.json")


if __name__ == "__main__":
    main()
