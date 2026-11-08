"""
Randomized starting hyperparameters for each model, the best ones found are then written in best_params.json.

Usage: python tune.py [n_iter] (default n_iter = 30)

Tuning uses a different CV split (seed 7) than the evaluation in train.py (seed 42),
so train.py's scores are less optimistic.
"""
import json
import sys
import time

import numpy as np
from scipy.stats import loguniform, randint, uniform
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from features import load_jsonl, make_features
from train import get_models

tune_seed = 7

# Hyperparameter search spaces for each model
spaces = {
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

# Convert numpy types to normal for the creation of the final json file
def to_json(v):
    return v.item() if isinstance(v, np.generic) else v


def main():

    # Loading, extracting and setting for CV
    n_iter = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    train = load_jsonl("train.jsonl")
    test = load_jsonl("test.jsonl")
    X, _ = make_features(train, test, "type_chart_gen2.csv")
    y = np.array([b["player_won"] for b in train])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=tune_seed)

    # Retrieving models at their untuned baseline (made on cpu because on gpu was running slower than on cpu)
    # and parallellized over search candidates rather than inside each model
    models = get_models(tuned=False, device="cpu", xgb_seeds=1)
    for name in ("random_forest", "extra_trees", "xgboost"):
        if name in models:
            models[name].set_params(n_jobs=1)


    best = {}
    for name, model in models.items():
        t0 = time.time()

        # Initizlized search on every model keeping track of accuracy and nll, refitting basing on nll and fitted on training data
        search = RandomizedSearchCV(
            model, spaces[name], n_iter=n_iter, cv=cv, n_jobs=-1, random_state=tune_seed,
            scoring={"logloss": "neg_log_loss", "acc": "accuracy"}, refit="logloss")
        search.fit(X, y)

        # Gets the best index and accuracy for that model, then
        # saves its hyerparameters in the dictionary made before cleaning the numpy types
        i = search.best_index_
        acc = search.cv_results_["mean_test_acc"][i]
        best[name] = {k: to_json(v) for k, v in search.best_params_.items()}
        print(f"{name:15s} logloss {-search.best_score_:.4f} acc {acc:.4f} "
              f"({time.time() - t0:.0f}s) {best[name]}", flush=True)

    # Creation of the final json file
    with open("best_params.json", "w") as f:
        json.dump(best, f, indent=2)
    print("Wrote best_params.json")


if __name__ == "__main__":
    main()
