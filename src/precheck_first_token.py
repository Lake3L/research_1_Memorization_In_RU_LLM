"""Which datasets can the first-token test be run on at all?

PREREGISTRATION.md §5 runs the first-token test "only where the row-independence
pre-check passes". The pre-check is the library's own: `first_token_test` calls
`statistical_feature_prediction_test` on the *first* feature — a gradient
boosted tree and a linear model trained to predict that feature of row n from
the rows before it — and refuses to run when the null of no dependence is
rejected. This script runs exactly that gate offline, seeded and over several
seeds (the library draws its test rows with an unseeded `np.random.choice`),
so that a GPU session is not spent on a cell the library will refuse.

Three outcomes per file:
  pass    — the first feature is not predictable from the preceding rows;
  reject  — it is (a sequential id, a file sorted by that column);
  error   — the check itself cannot run: a near-unique string first feature
            (an address, a domain, a product name) leaves the classifier with
            classes it never saw, and the library's ten retries all fail.
Only `pass` cells are schedulable. `--whole-file` additionally runs the
library's stricter `row_independence_test`, every feature with Bonferroni,
for the record; it is slow (XGBoost per feature) and is not the gate.

Usage:
  python src/precheck_first_token.py --group ru_pre_cutoff,fresh_control,canon
  python src/precheck_first_token.py --whole-file --seeds 42
"""

import argparse
import contextlib
import io
import json
import os
import sys
import time
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset_registry import load_registry  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def first_feature_gate(path, feature, seed):
    from tabmemcheck.row_independence import statistical_feature_prediction_test
    np.random.seed(seed)
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            rejected = statistical_feature_prediction_test(path, feature, confidence_level=0.99)
        return "reject" if rejected else "pass", None
    except Exception as e:  # tenacity wraps the last ValueError in a RetryError
        inner = getattr(e, "last_attempt", None)
        message = str(inner.exception()) if inner else str(e)
        return "error", message.splitlines()[0][:160]


def whole_file(path, seed):
    from tabmemcheck.row_independence import row_independence_test
    np.random.seed(seed)
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            rejected = row_independence_test(path, confidence_level=0.99)
        return "dependent" if rejected else "not rejected", None
    except Exception as e:
        inner = getattr(e, "last_attempt", None)
        message = str(inner.exception()) if inner else str(e)
        return "error", message.splitlines()[0][:160]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="ru_pre_cutoff,fresh_control,canon")
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--whole-file", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    from tabmemcheck import utils

    seeds = [int(s) for s in args.seeds.split(",")]
    groups = set(args.group.split(","))
    registry = load_registry()
    rows = []
    label = "whole-file row_independence_test" if args.whole_file else "first-feature gate of first_token_test"
    print(f"{label}, seeds {seeds}, confidence 0.99\n")
    print(f"{'dataset':26s} {'first feature':24s} verdicts")
    print("-" * 80)
    for name, rec in registry.items():
        if rec["group"] not in groups:
            continue
        path = os.path.join(ROOT, rec["variants"]["raw"]["path"])
        if not os.path.exists(path):
            print(f"{name:26s} (file missing — run src/fetch_data.py)")
            continue
        feature = utils.get_feature_names(path)[0]
        started = time.time()
        verdicts, notes = [], []
        for seed in seeds:
            verdict, note = whole_file(path, seed) if args.whole_file else first_feature_gate(path, feature, seed)
            verdicts.append(verdict)
            if note:
                notes.append(note)
        schedulable = (not args.whole_file) and all(v == "pass" for v in verdicts)
        rows.append({"dataset": name, "group": rec["group"], "first_feature": feature,
                     "check": label, "seeds": seeds, "verdicts": verdicts,
                     "first_token_schedulable": schedulable,
                     "note": notes[0] if notes else None,
                     "seconds": round(time.time() - started, 1)})
        print(f"{name:26s} {feature[:24]:24s} {verdicts}  ({time.time() - started:.0f}s)"
              + (f"\n{'':52s}{notes[0]}" if notes else ""))

    if not args.whole_file:
        ok = [r["dataset"] for r in rows if r["first_token_schedulable"]]
        print(f"\nfirst token schedulable on: {ok if ok else 'none'}")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)
        print("wrote", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
