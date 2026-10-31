"""Cross-validate several classic ML models on the battle features and write a submission.

Usage: python train.py
"""
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features import load_jsonl, make_features

SEED = 42
N_FOLDS = 5


def get_models():
    models = {
        "logreg": make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000)),
        "random_forest": RandomForestClassifier(
            n_estimators=800, min_samples_leaf=3, max_features=0.3, n_jobs=-1, random_state=SEED),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=800, min_samples_leaf=3, max_features=0.4, n_jobs=-1, random_state=SEED),
        "hist_gbm": HistGradientBoostingClassifier(
            learning_rate=0.03, max_iter=600, max_leaf_nodes=15, min_samples_leaf=40,
            l2_regularization=1.0, random_state=SEED),
    }
    try:
        from importlib import import_module
        XGBClassifier = import_module("xgboost").XGBClassifier
        models["xgboost"] = XGBClassifier(
            n_estimators=800, learning_rate=0.02, max_depth=4, subsample=0.8, colsample_bytree=0.6,
            min_child_weight=5, reg_lambda=2.0, n_jobs=-1, random_state=SEED, eval_metric="logloss")
    except ImportError:
        print("xgboost not installed, skipping it (pip install xgboost to enable)")
    return models


def main():
    train = load_jsonl("train.jsonl")
    test = load_jsonl("test.jsonl")
    X, X_test = make_features(train, test, "type_chart_gen2.csv")
    y = np.array([b["player_won"] for b in train])
    print(f"{X.shape[1]} features, {len(X)} train battles, {len(X_test)} test battles\n")

    models = get_models()
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = {name: np.zeros(len(X)) for name in models}
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        for name, model in models.items():
            m = clone(model).fit(X.iloc[tr_idx], y[tr_idx])
            oof[name][va_idx] = m.predict_proba(X.iloc[va_idx])[:, 1]
        print(f"fold {fold + 1}/{N_FOLDS} done")

    print(f"\n{'model':15s} {'acc':>7s} {'auc':>7s}")
    for name, p in oof.items():
        print(f"{name:15s} {accuracy_score(y, p > 0.5):7.4f} {roc_auc_score(y, p):7.4f}")
    ens = np.mean(list(oof.values()), axis=0)
    print(f"{'ensemble(mean)':15s} {accuracy_score(y, ens > 0.5):7.4f} {roc_auc_score(y, ens):7.4f}")

    # Feature importance from a random forest fit on all training data.
    rf = clone(models["random_forest"]).fit(X, y)
    imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("\nTop 20 features (random forest impurity importance):")
    print(imp.head(20).to_string())

    # Fit every model on all training data and average their test probabilities.
    test_pred = np.mean([clone(m).fit(X, y).predict_proba(X_test)[:, 1] for m in models.values()], axis=0)
    sub = pd.DataFrame({"battle_id": [b["battle_id"] for b in test],
                        "player_won": (test_pred > 0.5).astype(int)})
    sub.to_csv("submission.csv", index=False)
    print(f"\nWrote submission.csv ({len(sub)} rows, predicted win rate {sub.player_won.mean():.3f})")


if __name__ == "__main__":
    main()
