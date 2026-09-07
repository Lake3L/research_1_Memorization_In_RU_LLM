"""Score the first-token test against baselines that are properties of the data.

`tabmemcheck.functions.first_token_test` prints a line labelled "Baseline (Matches
of most common first token)" and computes it as

    np.sum(np.array(response_tokens) == most_common_first_token)

— the number of times *the model's own answers* were the most common token. That
is a description of the model's output habits, not a baseline. A baseline is what
a trivial predictor would score against the ground truth, and it cannot depend on
which model is being tested. Because it does, the same dataset returns different
baselines for different models: on adult-train the printed value was 73/250 for
one model and 19/250 for the other, while the correct value is 55/250 for both,
and a rate of 52/250 was consequently reported as significant at p = 3e-11 when it
is in fact below baseline.

This recomputes the test from the raw call log. Nothing has to be re-run: the
prompts and responses are in the log, and the ground truth follows from the
prompts.

Two families of baseline are computed, and AMENDMENT_6 §1 says which decides.

*Visible-information predictors* use only what the model was given: the prompt
(eight preceding rows) and the dataset's marginal distribution. `mode` always
answers the most common first token; `previous_row` repeats the first token of
the last prefix row, which is what a model would do if the file were sorted or
locally clustered on its first column. The decision baseline is the better of
the two.

*Row-conditional predictors* — logistic regression and gradient boosting fitted
on the observation's other features, with the column the token comes from
excluded — are what PREREGISTRATION.md §5 names. They are reported as a
sensitivity bound. They are not the decision baseline for this test, because
they read the target row's other features, which the first-token prompt does not
contain: the model is asked to continue the file, not to complete a row.

Usage:
  python src/rescore_first_token.py results/calls_*.jsonl
  python src/rescore_first_token.py results/calls_*.jsonl --out corrected.json
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset_registry import load_registry  # noqa: E402
from rescore_calls import block_index, prompt_text  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def first_token_of(text, num_digits):
    """What tabmemcheck compares: the first characters of the first answer line."""
    return str(text).strip("\n").split("\n")[0][:num_digits]


def binomial_p(matches, n, rate):
    if not n:
        return None
    return float(stats.binomtest(matches, n, min(max(rate, 1e-9), 1 - 1e-9),
                                 alternative="greater").pvalue)


def learned_baselines(df, tokens, seed=42):
    """Accuracy of logistic regression and gradient boosting at predicting the token.

    Cross-validated on the dataset. The first column is dropped: the token is its
    leading digits, so a model given it would score perfectly and the baseline
    would be meaningless.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OrdinalEncoder, StandardScaler
    from sklearn.impute import SimpleImputer

    features = df.iloc[:, 1:].copy()
    for column in features.columns:
        if features[column].dtype == object:
            features[column] = OrdinalEncoder(
                handle_unknown="use_encoded_value", unknown_value=-1
            ).fit_transform(features[[column]])
    X = SimpleImputer(strategy="most_frequent").fit_transform(features)
    y = np.asarray(tokens)

    counts = pd.Series(y).value_counts()
    keep = counts[counts >= 5].index
    mask = pd.Series(y).isin(keep).to_numpy()
    if mask.sum() < 25 or len(keep) < 2:
        return {}
    X, y = X[mask], y[mask]

    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    out = {}
    for name, model in (
        ("logistic_regression",
         make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))),
        ("gradient_boosting", HistGradientBoostingClassifier(random_state=seed)),
    ):
        try:
            out[name] = float(cross_val_score(model, X, y, cv=folds,
                                              scoring="accuracy").mean())
        except Exception:
            out[name] = None
    return out


def rescore(log_path, registry):
    from tabmemcheck import analysis, utils

    calls = [json.loads(l) for l in open(log_path, encoding="utf-8")]
    cells = {}
    for call in calls:
        if call.get("test") != "first_token":
            continue
        cells.setdefault(call.get("dataset"), []).append(call)

    results = []
    for dataset, group in cells.items():
        record = registry.get(dataset[:-4])
        if record is None:
            continue
        path = os.path.join(ROOT, record["variants"]["raw"]["path"])
        # the prompts were built from the header-inclusive row list, so the
        # block index must use it too; the tokens and the frame must not, since
        # the header is not an observation and tabmemcheck excludes it
        index = block_index(utils.load_csv_rows(path))
        num_digits = analysis.build_first_token(path)
        df = utils.load_csv_df(path)
        data_rows = utils.load_csv_rows(path, header=False)
        all_tokens = [r[:num_digits] for r in data_rows]
        mode_token = pd.Series(all_tokens).value_counts().index[0]

        truth, answers, previous = [], [], []
        for call in group:
            prompt = prompt_text(call).strip()
            true_row = index.get(prompt)
            if true_row is None:
                continue
            truth.append(true_row[:num_digits])
            answers.append(first_token_of(call["response"], num_digits))
            previous.append(prompt.split("\n")[-1][:num_digits])

        n = len(truth)
        matches = sum(1 for t, a in zip(truth, answers) if t == a)
        visible = {
            "mode": sum(1 for t in truth if t == mode_token) / n if n else 0.0,
            "previous_row": (sum(1 for t, p in zip(truth, previous) if t == p) / n
                             if n else 0.0),
        }
        learned = {k: v for k, v in learned_baselines(df, all_tokens).items()
                   if v is not None}
        baseline_source = max(visible, key=visible.get)
        baseline = visible[baseline_source]
        conditional = max(learned.values()) if learned else None
        library = sum(1 for a in answers if a == mode_token)

        results.append({
            "dataset": dataset, "n": n, "matches": matches,
            "rate": matches / n if n else 0.0,
            "num_digits": num_digits, "mode_token": mode_token,
            "baselines": {**visible, **learned},
            "baseline": baseline, "baseline_source": baseline_source,
            "p_value": binomial_p(matches, n, baseline),
            "row_conditional_baseline": conditional,
            "row_conditional_source": (max(learned, key=learned.get) if learned else None),
            "p_value_row_conditional": (binomial_p(matches, n, conditional)
                                        if conditional is not None else None),
            "library_reported_baseline": library / n if n else None,
        })
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    registry = load_registry()
    paths = []
    for pattern in args.logs:
        paths.extend(sorted(glob.glob(pattern)) or [pattern])

    everything = {}
    header = (f"{'dataset':22s} {'result':>9s} {'rate':>6s} {'baseline':>9s} "
              f"{'from':>13s} {'p':>9s} {'row-cond.':>9s} {'p':>9s} {'library':>8s}")
    for path in paths:
        rows = rescore(path, registry)
        if not rows:
            continue
        everything[os.path.basename(path)] = rows
        print("=" * len(header))
        print(os.path.basename(path)[:len(header)])
        print(header)
        print("-" * len(header))
        for r in rows:
            cond = r["row_conditional_baseline"]
            print(f"{r['dataset']:22s} {r['matches']:4d}/{r['n']:<4d} {r['rate']:6.3f} "
                  f"{r['baseline']:9.3f} {r['baseline_source']:>13s} {r['p_value']:9.2e} "
                  + (f"{cond:9.3f} {r['p_value_row_conditional']:9.2e} " if cond is not None
                     else f"{'-':>9s} {'-':>9s} ")
                  + f"{r['library_reported_baseline']:8.3f}")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(everything, f, ensure_ascii=False, indent=2)
        print("\nwrote", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
