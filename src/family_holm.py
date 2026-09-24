"""The H2e family of AMENDMENT_9, with the Holm correction of §5.

Collects every model × dataset × test p-value of the high-exposure tables
from the committed scoring files, applies Holm within the family at α = 0.05,
and states each cell's verdict:

  row       AMENDMENT_7: p over the queries R1 and R4 keep, against the R2
            null; positive only if a witness absent from the prompt was also
            reproduced (R3). From results/prefix_baseline_<run>.json.
  feature   §5: exact matches against the conditional baseline (best of mode /
            LR / GBT / 1-NN, data/feature_baselines.json). From
            results/classifier_diagnostics_<run>.json.

A negative cell in which fewer than half the row answers even have the shape
of a CSV row is reported **inconclusive**, not negative: it cannot tell a
model that does not remember from one that did not answer. This is the
FAIL_ADAPTER rule of the block A gate (reports/RESULTS_GATE.md §6, written
before that run), applied cell by cell and only to negatives — a significant
cell consists of row-shaped matches by construction. Inconclusive cells stay
in the family, which only makes the correction stricter for the rest. The
header test gives a verdict, not a p-value (§5), and is listed apart.

Usage:
  python src/family_holm.py --prefix results/prefix_baseline_exposure_1_*.json results/prefix_baseline_exposure_2_*.json
                            --diagnostics results/classifier_diagnostics_exposure_*.json
                            --results "results/gateA_exposure_*.json"
                            --out results/h2e_family_<stamp>.json
"""

import argparse
import glob
import json
import os

ALPHA = 0.05
WELL_FORMED_FLOOR = 0.5


def holm(pvalues):
    """Holm-adjusted p-values, in the input order."""
    order = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    m = len(pvalues)
    adjusted, running = [0.0] * m, 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvalues[i]))
        adjusted[i] = running
    return adjusted


def short(model):
    return "adapted" if "vikhr" in model.lower() else "base"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", nargs="+", required=True)
    ap.add_argument("--diagnostics", nargs="+", required=True)
    ap.add_argument("--results", nargs="+", required=True)
    ap.add_argument("--group", default="ru_exposure")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from dataset_registry import load_registry
    members = {f"{n}.csv" for n, r in load_registry().items() if r["group"] == args.group}

    well_formed, header = {}, []
    for pattern in args.results:
        for path in sorted(glob.glob(pattern)):
            run = json.load(open(path, encoding="utf-8"))
            for c in run["results"]:
                if c["dataset_key"] not in members:
                    continue
                if c["test"] == "row":
                    well_formed[(run["model"], c["dataset_key"])] = c.get("well_formed_rate")
                elif c["test"] == "header":
                    header.append({"model": run["model"], "dataset": c["dataset_key"],
                                   "verdict": c.get("verdict") or c.get("header_verdict")
                                   or ("pass" if c.get("passed") else "fail")})

    cells = []
    for path in args.prefix:
        for key, s in json.load(open(path, encoding="utf-8"))["cells"].items():
            model, dataset = key.split("|")
            if dataset not in members:
                continue
            p = s["after_rules"]["p"]
            cells.append({"model": model, "dataset": dataset, "test": "row",
                          "count": f"{s['after_rules']['matches']}/{s['after_rules']['n']}",
                          "p": 1.0 if p is None else p,
                          "witness": f"{s['witness']['reproduced']}/{s['witness']['present']}",
                          "witness_ok": s["witness"]["reproduced"] >= 1,
                          "well_formed": well_formed.get((model, dataset))})
    for path in args.diagnostics:
        for model, entry in json.load(open(path, encoding="utf-8"))["models"].items():
            for f in entry["feature"]:
                if f["dataset"] not in members:
                    continue
                cells.append({"model": model, "dataset": f["dataset"], "test": "feature",
                              "count": f"{f['matches']}/{f['n']}", "p": f["p"],
                              "baseline": f"{f['baseline']:.4f} [{f['baseline_from']}]",
                              "witness_ok": True, "well_formed": None})

    for c, adj in zip(cells, holm([c["p"] for c in cells])):
        c["p_holm"] = adj
        significant = adj < ALPHA
        if significant and c["witness_ok"]:
            c["verdict"] = "POSITIVE"
        elif (c["test"] == "row" and c["well_formed"] is not None
              and c["well_formed"] < WELL_FORMED_FLOOR):
            c["verdict"] = "inconclusive"
        else:
            c["verdict"] = "negative"

    cells.sort(key=lambda c: (c["dataset"], c["test"], short(c["model"])))
    print(f"H2e family: {len(cells)} p-values, Holm at alpha = {ALPHA}\n")
    print(f"{'dataset':28s} {'test':8s} {'model':8s} {'count':>9s} {'p':>10s} {'p (Holm)':>10s} "
          f"{'witness / baseline':>22s} {'wf':>5s}  verdict")
    for c in cells:
        wf = f"{c['well_formed']:.0%}" if isinstance(c["well_formed"], float) else "-"
        extra = c.get("witness") or c.get("baseline", "")
        print(f"{c['dataset']:28s} {c['test']:8s} {short(c['model']):8s} {c['count']:>9s} "
              f"{c['p']:10.2e} {c['p_holm']:10.2e} {extra:>22s} {wf:>5s}  {c['verdict']}")
    print("\nheader test (verdicts, outside the family):")
    for h in sorted(header, key=lambda h: (h["dataset"], short(h["model"]))):
        print(f"  {h['dataset']:28s} {short(h['model']):8s} {h['verdict']}")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"_what": __doc__.split("\n\n")[0], "alpha": ALPHA,
                   "well_formed_floor": WELL_FORMED_FLOOR, "cells": cells, "header": header},
                  f, ensure_ascii=False, indent=2)
    print("\nwrote", args.out)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    main()
