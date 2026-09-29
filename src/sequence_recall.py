"""Does a model continue a classifier it holds, or one it reconstructs?

Exploratory, cross-model; no verdict of AMENDMENT_7, §5 or family_holm.py
depends on it (reports/RESULTS_EXPOSURE.md Part V). It extends the gradient
of classifier_diagnostics.py — row completion by the distance of the true
row's name from the closest name among the prompt rows, bins frozen there
before any model other than the Nemo pair ran — in three ways:

  bins        the library's whole-row match, and the name field alone, per
              bin, for every call log given. A model that reconstructs the
              next row from its neighbours matches where the next name is a
              variation of theirs and almost never beyond 0.5; a model that
              holds the classification as ordered text names the next
              rubric at any distance.
  paired      exact McNemar over the identical prompts of two logs, within
              one bin, for each pair of labels given with --pair.
  file_fields on МКБ-10 only, the fields that belong to the portal's file and
              not to the classification — the parent record's id and the
              four-character prefix of the record code — in the queries where
              the prompt does not show the true value. Reproducing them there
              would be memory of the file; reproducing only the names is
              memory of the classification.

Usage:
  python src/sequence_recall.py --log LABEL=results/calls_exposure_1_<model>_<stamp>.jsonl [...]
                                --pair LABEL_A:LABEL_B [...]
                                --out results/sequence_recall_<stamp>.json
"""

import argparse
import csv
import json
import os
import sys

from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from classifier_diagnostics import BINS, NAME_FIELD, calls_of, path_of  # noqa: E402
from prefix_baseline import block_index, library_match, load_rows, norm_lev  # noqa: E402

DATASETS = ("mkb10_v2.csv", "okved2.csv")
# МКБ-10 export: column 1 is the record code ("1409N760": four characters of
# chapter and block, then the code), column 4 the parent record's id
FILE_FIELDS = {"mkb10_v2.csv": {"parent_id": (4, None), "record_code_prefix": (1, 4)}}


def fields(line):
    try:
        return next(csv.reader([line]))
    except (csv.Error, StopIteration):
        return []


def field(values, column, width=None):
    if len(values) <= column:
        return None
    value = values[column].strip()
    return value[:width] if width else value


def queries(path, dataset):
    """Per row-completion query of `dataset`: prompt, true next row, answer."""
    index = block_index(load_rows(path_of(dataset)))
    out = {}
    for c in calls_of(path):
        if c.get("test") != "row" or c.get("kind") != "prompt" or c["dataset"] != dataset:
            continue
        located = index.get(c["prompt"].strip())
        if located:
            out[c["prompt"].strip()] = (located[1], c["response"])
    return out


def score(prompt, truth, response, dataset):
    j = NAME_FIELD[dataset]
    rows = prompt.split("\n")
    name = field(fields(truth), j) or ""
    distance = min(norm_lev(name, field(fields(r), j) or "") for r in rows)
    first = response.strip("\n").split("\n")[0] if response.strip() else ""
    answer = fields(first)
    entry = {"distance": distance, "row": bool(library_match(truth, response)),
             "name": field(answer, j) == name}
    for key, (column, width) in FILE_FIELDS.get(dataset, {}).items():
        shown = {field(fields(r), column, width) for r in rows}
        if key == "parent_id":  # an id shown as a record's own id counts as shown
            shown |= {field(fields(r), 0) for r in rows}
        true_value = field(fields(truth), column, width)
        entry[key] = None if true_value in shown else field(answer, column, width) == true_value
    return entry


def bin_of(distance):
    return next(b for b in BINS if b[0] <= distance < b[1])


def label_of(b):
    return f"[{b[0]:.2f}, {b[1]:.2f})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", action="append", required=True, metavar="LABEL=PATH")
    ap.add_argument("--pair", action="append", default=[], metavar="A:B")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    logs = dict(item.split("=", 1) for item in args.log)

    result = {"_what": __doc__.split("\n\n")[0], "logs": logs, "datasets": {}}
    for dataset in DATASETS:
        scored = {}
        for label, path in logs.items():
            scored[label] = {p: score(p, t, r, dataset) for p, (t, r) in queries(path, dataset).items()}
        common = set.intersection(*(set(s) for s in scored.values()))
        entry = {"common_prompts": len(common), "bins": {}, "paired": [], "file_fields": {}}

        print(f"\n{dataset}: {len(common)} prompts common to all logs")
        print("  whole row / name alone, by distance of the next name to the closest prompt name")
        for label, s in scored.items():
            cells = {label_of(b): {"n": 0, "row": 0, "name": 0} for b in BINS}
            for p in common:
                c = cells[label_of(bin_of(s[p]["distance"]))]
                c["n"] += 1
                c["row"] += s[p]["row"]
                c["name"] += s[p]["name"]
            entry["bins"][label] = cells
            print(f"    {label:10s} " + "  ".join(
                f"{b} {v['row']:>3d}|{v['name']:<3d}/{v['n']:<3d}" for b, v in cells.items()))

        for pair in args.pair:
            a, b = pair.split(":")
            for bn in BINS:
                ps = [p for p in common if bin_of(scored[a][p]["distance"]) == bn]
                for measure in ("row", "name"):
                    x = sum(scored[a][p][measure] and not scored[b][p][measure] for p in ps)
                    y = sum(scored[b][p][measure] and not scored[a][p][measure] for p in ps)
                    pv = float(stats.binomtest(x, x + y, 0.5).pvalue) if x + y else 1.0
                    entry["paired"].append({"a": a, "b": b, "bin": label_of(bn), "measure": measure,
                                            "n": len(ps), "a_total": sum(scored[a][p][measure] for p in ps),
                                            "b_total": sum(scored[b][p][measure] for p in ps),
                                            "a_only": x, "b_only": y, "p": pv})
            far = [e for e in entry["paired"] if e["a"] == a and e["b"] == b and e["bin"] == label_of(BINS[-1])]
            for e in far:
                print(f"  paired {a} vs {b}, bin {e['bin']}, {e['measure']:4s}: {e['a_total']} vs {e['b_total']} "
                      f"of {e['n']}; discordant {e['a_only']}/{e['b_only']}, McNemar p = {e['p']:.2g}")

        for key in FILE_FIELDS.get(dataset, {}):
            for label, s in scored.items():
                hidden = [s[p] for p in common if s[p][key] is not None]
                entry["file_fields"].setdefault(key, {})[label] = {
                    "hidden": len(hidden), "field_right": sum(h[key] for h in hidden),
                    "name_right": sum(h["name"] for h in hidden)}
            print(f"  {key} not shown by the prompt: " + "; ".join(
                f"{label} {v['field_right']}/{v['hidden']} (name {v['name_right']})"
                for label, v in entry["file_fields"][key].items()))
        result["datasets"][dataset] = entry

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print("\nwrote", args.out)


if __name__ == "__main__":
    main()
