"""Quality gate every dataset must pass before it enters the study.

A dataset is only useful here if the classification task it defines is actually
learnable. If logistic regression and gradient boosting cannot beat the majority
class, then an LLM's few-shot accuracy on it carries no signal either, and the
dataset cannot support H3 — it would look identical whether the model memorized
it or not.

The baselines double as the reference numbers H3 needs anyway (the LLM is
compared against them), and as the check that the four dataset formats leave the
classification problem intact, which is how Bordt et al. validate their
transforms in their Tables 9-10.

Cross-validation is stratified and shuffled on purpose: these files arrive
grouped by region or by class, and unshuffled folds silently measure
generalisation across regions instead of accuracy.
"""

import argparse
import json
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def baselines(df: pd.DataFrame, target: str, seed: int = 42) -> dict:
    y = df[target]
    X = df.drop(columns=[target])
    categorical = [c for c in X.columns if X[c].dtype == object]
    numeric = [c for c in X.columns if c not in categorical]
    X = X.copy()
    for c in categorical:
        X[c] = X[c].fillna("__na__").astype(str)

    # Numeric gaps are common in these sources (a city with no founding year on
    # record). Median imputation keeps the row instead of discarding it, which
    # matters when the gaps are not random across classes.
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", max_categories=100,
                              sparse_output=False), categorical),
        ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()),
         numeric),
    ])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    out = {}
    for name, clf in [("logistic_regression", LogisticRegression(max_iter=3000)),
                      ("gradient_boosting", HistGradientBoostingClassifier(random_state=seed))]:
        pred = cross_val_predict(make_pipeline(pre, clf), X, y, cv=cv)
        acc = float((pred == y).mean())
        out[name] = {
            "accuracy": round(acc, 4),
            "cohen_kappa": round(float(cohen_kappa_score(y, pred)), 4),
        }
    majority = float(max(y.mean(), 1 - y.mean())) if y.nunique() == 2 else \
        float(y.value_counts(normalize=True).max())
    out["majority_baseline"] = round(majority, 4)
    out["best_lift_over_majority"] = round(
        max(out["logistic_regression"]["accuracy"],
            out["gradient_boosting"]["accuracy"]) - majority, 4)
    return out


def profile(df: pd.DataFrame, target: str) -> dict:
    rows = df.astype(str).apply(lambda r: ",".join(r), axis=1)
    return {
        "n_rows": len(df),
        "n_cols": df.shape[1],
        "duplicate_row_share": round(1 - rows.nunique() / len(rows), 4),
        "missing_by_column": {c: round(float(df[c].isna().mean()), 3)
                              for c in df.columns if df[c].isna().any()},
        "cardinality": {c: int(df[c].nunique()) for c in df.columns},
        "class_balance": round(float(df[target].mean()), 4)
        if df[target].nunique() == 2 else None,
    }


def verdict(prof: dict, base: dict) -> dict:
    """Preregistered admission criteria (PREREGISTRATION.md §4)."""
    checks = {
        "at_least_300_rows": prof["n_rows"] >= 300,
        "task_is_learnable": base["best_lift_over_majority"] >= 0.05,
        "not_dominated_by_duplicates": prof["duplicate_row_share"] < 0.25,
        "has_high_entropy_feature": any(
            v / prof["n_rows"] > 0.5 for k, v in prof["cardinality"].items()),
    }
    return {"checks": checks, "admitted": all(checks.values())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--target", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    prof = profile(df, args.target)
    base = baselines(df, args.target)
    report = {"file": os.path.relpath(args.csv, ROOT), "target": args.target,
              "profile": prof, "baselines": base, **verdict(prof, base)}

    print(json.dumps(report, ensure_ascii=False, indent=2))
    out = args.out or args.csv.replace(".csv", "_quality.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
