"""What would our tests have caught? The minimum detectable effect, per dataset.

A zero is not a finding until you know what a non-zero would have taken. This is
the difference between "the model did not memorize this dataset" and "our test
could not have told us either way", and for the Russian datasets it is the whole
problem: the canon has ground truth — Bordt et al. established that GPT-4
reproduces iris — while for `mos_torgovye_obekty` nobody knows what the answer
should be. That is the point of testing them, and it is also why a null there is
uninterpretable unless the test's sensitivity is stated alongside it.

So every reported zero carries the smallest rate the cell could have rejected the
baseline at, given the number of queries actually run. Two dataset properties
drive it:

  the baseline — row completion is scored against the duplicate-row rate, because
    reproducing a row that occurs twice is not evidence of anything. A dataset
    with many duplicates has a high bar before a count means memorization.
  the ceiling  — the test cannot ask more questions than the file has rows, so
    small datasets are capped no matter how much compute is available.

Row length in digits is reported next to both, as Ward et al.'s covariate: long
digit strings are harder to reproduce, so the same rate on two datasets is not
the same feat, and a Russian table with long identifiers is a harder target than
iris regardless of what any model saw.

Usage:
  python src/detectability.py
  python src/detectability.py --group ru_pre_cutoff --queries 250
"""

import argparse
import json
import math
import os
import sys

from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PREFIX_ROWS = 8


def minimum_detectable_rate(baseline, n, alpha=0.05, power=0.8):
    """Smallest true rate a one-sided binomial test would call at this n.

    Found by search rather than by a normal approximation, because the baselines
    here are tiny (often zero duplicates) and the normal approximation is poor
    exactly there.
    """
    if n <= 0:
        return None
    # the count a one-sided exact test would need to reject at alpha
    critical = None
    for k in range(n + 1):
        if stats.binomtest(k, n, max(baseline, 1e-9),
                           alternative="greater").pvalue < alpha:
            critical = k
            break
    if critical is None:
        return None
    # the true rate at which we would reach that count with `power` probability
    low, high = baseline, 1.0
    for _ in range(60):
        mid = (low + high) / 2
        if stats.binom.sf(critical - 1, n, mid) < power:
            low = mid
        else:
            high = mid
    return high


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default=None,
                    help="canon | ru_pre_cutoff | fresh_control; default: all")
    ap.add_argument("--queries", type=int, default=250,
                    help="queries per cell, before the dataset ceiling is applied")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    registry = json.load(open(os.path.join(ROOT, "data", "registry.json"),
                              encoding="utf-8"))
    rows = []
    for name, rec in sorted(registry.items(), key=lambda kv: kv[1]["group"]):
        if args.group and rec["group"] != args.group:
            continue
        diagnostics = rec["diagnostics"]
        baseline = diagnostics["duplicate_row_share"]
        ceiling = max(rec["n_rows"] - PREFIX_ROWS, 0)
        n = min(args.queries, ceiling)
        mdr = minimum_detectable_rate(baseline, n)
        rows.append({
            "dataset": name, "group": rec["group"], "n_rows": rec["n_rows"],
            "duplicate_baseline": baseline, "query_ceiling": ceiling,
            "queries_used": n,
            "minimum_detectable_rate": round(mdr, 4) if mdr else None,
            "digits_per_row": diagnostics["mean_digits_per_row"],
            "chars_per_row": diagnostics["mean_chars_per_row"],
        })

    print(f"Row completion, {args.queries} queries requested, "
          f"{PREFIX_ROWS} prefix rows, one-sided exact binomial at alpha 0.05, "
          f"power 0.80\n")
    header = (f"{'dataset':26s} {'group':14s} {'dup base':>9s} {'ceiling':>8s} "
              f"{'n used':>7s} {'min detectable':>14s} {'digits/row':>11s}")
    print(header)
    print("-" * len(header))
    for r in rows:
        mdr = (f"{r['minimum_detectable_rate']:.1%}"
               if r["minimum_detectable_rate"] else "—")
        print(f"{r['dataset']:26s} {r['group']:14s} {r['duplicate_baseline']:9.4f} "
              f"{r['query_ceiling']:8d} {r['queries_used']:7d} {mdr:>14s} "
              f"{r['digits_per_row']:11.1f}")

    print("\nRead this as: a zero on that dataset excludes memorization rates above")
    print("the last column, and says nothing about anything below it. The comparison")
    print("that survives across datasets is rate-against-its-own-baseline, never a")
    print("raw count, and never a count across datasets with different row entropy.")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)
        print("wrote", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
