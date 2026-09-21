"""The conditional baseline of the feature-completion test (§5), computed from
the data alone, before the run it will judge.

§5 scores feature completion against "the conditional-baseline rate (best of
mode / LR / GBT predicting that feature)". The runner records only the mode
rate, because the decision rules are applied offline; this is the rest of it.
Per dataset and designated feature:

  mode   the share of the most common value — what a model that ignores the
         row can score;
  lr     logistic regression predicting the feature from every other column;
  gbt    gradient boosting, the same;
  nn1    one nearest neighbour over the same encoded columns, which is the
         predictor that *looks up* a similar row, and is the only conditional
         predictor that stays computable when the feature is a name with
         thousands of values.

The reported baseline is the largest of those computed, which is conservative:
a higher baseline makes a positive verdict harder, never easier. LR and GBT are
fitted only while the feature has at most `--max-classes` distinct values —
beyond that they are neither computable in reasonable time nor meaningful, and
the JSON says so rather than leaving a gap. Folds are stratified where every
class has at least two members and shuffled otherwise; the file's own order is
never used, since these files are sorted by code.

Usage:
  python src/feature_baseline.py --group ru_exposure --out data/feature_baselines.json
  python src/feature_baseline.py --only okved2,mkb10_v2
"""

import argparse
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset_registry import load_registry, read_table  # noqa: E402

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QUERIES = 250
SEED = 42


def features_of_plan():
    """The designated feature per dataset, from the runner, so that the baseline
    and the test cannot drift apart."""
    from run_repro import FEATURES
    return {k[:-4]: v for k, v in FEATURES.items()}


def encoded(X: pd.DataFrame):
    categorical = [c for c in X.columns if X[c].dtype == object]
    numeric = [c for c in X.columns if c not in categorical]
    X = X.copy()
    for c in categorical:
        X[c] = X[c].fillna("__na__").astype(str)
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", max_categories=100,
                              sparse_output=False), categorical),
        ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), numeric),
    ])
    return X, pre


def folds(y, seed=SEED, n=5):
    counts = y.value_counts()
    if counts.min() >= 2 and len(counts) <= len(y) / 2:
        return StratifiedKFold(n_splits=min(n, int(counts.min())), shuffle=True, random_state=seed)
    return KFold(n_splits=n, shuffle=True, random_state=seed)


def baseline_for(path: str, feature: str, max_classes: int) -> dict:
    df = read_table(path)
    if feature not in df.columns:
        return {"error": f"feature {feature!r} not in {list(df.columns)[:6]}…"}
    y = df[feature].astype(str).fillna("__na__")
    X = df.drop(columns=[feature])
    n_classes = int(y.nunique())
    mode = float(y.value_counts(normalize=True).iloc[0])
    out = {"n_rows": int(len(df)), "n_classes": n_classes, "mode": round(mode, 6)}

    Xe, pre = encoded(X)
    cv = folds(y)
    scores = {}
    try:
        pred = cross_val_predict(make_pipeline(pre, KNeighborsClassifier(n_neighbors=1)),
                                 Xe, y, cv=cv)
        scores["nn1"] = float((pred == y.values).mean())
    except Exception as e:  # noqa: BLE001
        scores["nn1"] = None
        out["nn1_note"] = f"{type(e).__name__}: {e}"[:120]
    if n_classes <= max_classes:
        for name, clf in [("lr", LogisticRegression(max_iter=3000)),
                          ("gbt", HistGradientBoostingClassifier(random_state=SEED))]:
            try:
                pred = cross_val_predict(make_pipeline(pre, clf), Xe, y, cv=cv)
                scores[name] = float((pred == y.values).mean())
            except Exception as e:  # noqa: BLE001
                scores[name] = None
                out[f"{name}_note"] = f"{type(e).__name__}: {e}"[:120]
    else:
        scores["lr"] = scores["gbt"] = None
        out["lr_gbt_note"] = (f"not fitted: {n_classes} classes exceed the cap of {max_classes}; "
                              "nn1 is the conditional predictor at this cardinality")
    out.update({k: (round(v, 6) if isinstance(v, float) else v) for k, v in scores.items()})
    computed = [v for v in [mode] + [scores.get(k) for k in ("lr", "gbt", "nn1")]
                if isinstance(v, float)]
    p0 = max(computed)
    out["baseline"] = round(p0, 6)
    out["baseline_from"] = max([("mode", mode)] + [(k, v) for k, v in scores.items()
                                                   if isinstance(v, float)], key=lambda kv: kv[1])[0]
    for k in range(QUERIES + 1):
        if stats.binomtest(k, QUERIES, p0, alternative="greater").pvalue < 0.05:
            out["minimum_detectable_rate_250"] = round(k / QUERIES, 4)
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default=None)
    ap.add_argument("--only", default=None)
    ap.add_argument("--max-classes", type=int, default=100)
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "feature_baselines.json"))
    args = ap.parse_args()

    registry = load_registry()
    features = features_of_plan()
    names = [n for n in registry
             if (not args.group or registry[n]["group"] == args.group)
             and (not args.only or n in args.only.split(","))
             and n in features]
    result = json.load(open(args.out, encoding="utf-8")) if os.path.exists(args.out) else {}
    result.pop("_what", None)

    print(f"{'dataset':24s} {'feature':34s} {'rows':>7s} {'classes':>8s} {'mode':>7s} "
          f"{'lr':>7s} {'gbt':>7s} {'nn1':>7s} {'baseline':>9s} {'MDR@250':>8s}")
    for name in names:
        path = os.path.join(ROOT, registry[name]["variants"]["raw"]["path"])
        b = baseline_for(path, features[name], args.max_classes)
        b["feature"] = features[name]
        b["group"] = registry[name]["group"]
        result[name] = b
        fmt = lambda v: f"{v:7.4f}" if isinstance(v, float) else f"{'-':>7s}"  # noqa: E731
        print(f"{name:24s} {features[name][:34]:34s} {b.get('n_rows', 0):7d} {b.get('n_classes', 0):8d} "
              f"{fmt(b.get('mode'))} {fmt(b.get('lr'))} {fmt(b.get('gbt'))} {fmt(b.get('nn1'))} "
              f"{fmt(b.get('baseline'))} {b.get('minimum_detectable_rate_250', float('nan')):8.3f}"
              + (f"   [{b['baseline_from']}]" if "baseline_from" in b else ""))
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"_what": "§5 conditional baseline of the feature-completion test: "
                                "best of mode / LR / GBT / 1-NN predicting the designated "
                                "feature from the other columns, 5-fold, computed before the "
                                "run it judges (src/feature_baseline.py)", **result},
                      f, ensure_ascii=False, indent=2)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
