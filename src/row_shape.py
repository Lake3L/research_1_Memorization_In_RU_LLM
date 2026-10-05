"""Is a row-completion answer shaped like a CSV row? Two readings, every cell.

The FAIL_ADAPTER rule (reports/RESULTS_GATE.md §6, extended cell by cell in
reports/RESULTS_EXPOSURE.md §11 and §20) calls a negative cell inconclusive
when fewer than half the answers even have the shape of a CSV row. The runner
measured that shape as `well_formed_rate` by counting delimiter characters in
the answer's first line against the true row (run_repro.response_diagnostics).
That count is right for files whose fields never contain the delimiter, and
wrong for files with quoted text: a row of russian_retail whose description
has seven commas is "not well-formed" against a true row whose description has
four, although both are rows of eleven fields. Session C2 (2026-10-03) found
it there: 3.6% by the count, 84% by parsing.

This script re-reads every row-completion answer of the given call logs and
reports both readings per cell:

  delimiter_count  the runner's measure, recomputed (must equal the recorded
                   well_formed_rate; a mismatch is printed)
  csv_fields       the first non-empty line parsed as CSV, quotes respected,
                   has as many fields as the true next row

and marks the cells where the two fall on different sides of the 50% floor.

Usage:
  python src/row_shape.py results/calls_<run>.jsonl [...] --out results/row_shape_<stamp>.json
"""

import argparse
import csv
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import infer_separator  # noqa: E402
from prefix_baseline import block_index, load_rows  # noqa: E402

FLOOR = 0.5


def first_line(text):
    """The line the runner and the library read: the first non-empty one."""
    for line in str(text).strip().split("\n"):
        if line.strip():
            return line.strip()
    return ""


def n_fields(line, sep):
    try:
        return len(next(csv.reader([line], delimiter=sep)))
    except (csv.Error, StopIteration):
        return 0


def cell_shapes(calls, path):
    rows = load_rows(path)
    index = block_index(rows)
    out = {"n": 0, "delimiter_count": 0, "csv_fields": 0, "unlocated": 0}
    for c in calls:
        located = index.get(c["prompt"].strip())
        if not located:
            out["unlocated"] += 1
            continue
        truth, got = located[1].strip(), first_line(c["response"])
        sep = infer_separator(truth)
        out["n"] += 1
        out["delimiter_count"] += truth.count(sep) > 0 and got.count(sep) == truth.count(sep)
        out["csv_fields"] += bool(got) and n_fields(got, sep) == n_fields(truth, sep)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("call_logs", nargs="+")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from dataset_registry import load_registry
    registry = load_registry()
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    paths = sorted({p for pattern in args.call_logs for p in glob.glob(pattern)})
    cells = []
    for log in paths:
        calls = [json.loads(line) for line in open(log, encoding="utf-8")]
        results_file = log.replace("calls_", "gateA_").replace(".jsonl", ".json")
        recorded = {}
        if os.path.exists(results_file):
            for r in json.load(open(results_file, encoding="utf-8"))["results"]:
                if r["test"] == "row":
                    recorded[r["dataset_key"]] = r.get("well_formed_rate")
        model = calls[0]["model"] if calls else "?"
        for dataset in sorted({c["dataset"] for c in calls if c.get("test") == "row"}):
            group = [c for c in calls if c.get("test") == "row" and c.get("kind") == "prompt"
                     and c["dataset"] == dataset]
            path = os.path.join(ROOT, registry[dataset[:-4]]["variants"]["raw"]["path"])
            s = cell_shapes(group, path)
            n = s["n"] or 1
            entry = {"log": os.path.basename(log), "model": model, "dataset": dataset,
                     "n": s["n"], "unlocated": s["unlocated"],
                     "delimiter_count_rate": round(s["delimiter_count"] / n, 4),
                     "csv_fields_rate": round(s["csv_fields"] / n, 4),
                     "recorded_well_formed_rate": recorded.get(dataset)}
            entry["recomputed_matches_recorded"] = (
                entry["recorded_well_formed_rate"] is None
                or abs(entry["recorded_well_formed_rate"] * len(group) / n
                       - entry["delimiter_count_rate"]) < 0.01)
            entry["sides_differ"] = ((entry["delimiter_count_rate"] < FLOOR)
                                     != (entry["csv_fields_rate"] < FLOOR))
            cells.append(entry)

    print(f"{'model':32s} {'dataset':30s} {'n':>4s} {'count':>6s} {'csv':>6s}  note")
    for e in cells:
        note = []
        if e["sides_differ"]:
            note.append("CROSSES THE 50% FLOOR")
        if not e["recomputed_matches_recorded"]:
            note.append(f"recorded {e['recorded_well_formed_rate']}")
        print(f"{e['model'].split('/')[-1][:32]:32s} {e['dataset'][:30]:30s} {e['n']:>4d} "
              f"{e['delimiter_count_rate']:6.0%} {e['csv_fields_rate']:6.0%}  {'; '.join(note)}")
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"_what": __doc__.split("\n\n")[0], "floor": FLOOR, "cells": cells},
                  f, ensure_ascii=False, indent=2)
    print("\nwrote", args.out)


if __name__ == "__main__":
    main()
