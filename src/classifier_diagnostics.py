"""Exploratory diagnostics of the exposure sessions (reports/RESULTS_EXPOSURE.md).

Not a decision rule: nothing here changes a verdict of AMENDMENT_7 or §5. It
answers the question AMENDMENT_9's positive left open — whether a classifier
row that a model reproduces is remembered, or reconstructed from the rows the
prompt already shows (Prashanth et al., arXiv:2406.17746, "reconstruction of
inherently predictable sequences") — and records three facts the verdicts
alone do not show. Four tables:

  gradient     row completion on a classifier: the match rate as a function of
               how far the true row's name is from the closest name among the
               eight prompt rows (normalised Levenshtein). A table the model
               has memorised is reproduced whatever the distance; a table it
               reconstructs is reproduced where the next name is a variation of
               the names it was shown.
  feature      feature completion against the conditional baseline of
               data/feature_baselines.json (best of mode / LR / GBT / 1-NN),
               one-sided exact binomial, with the library's own counts
               re-derived from the call log.
  portal_form  ОКСМ feature completion split by what only this table carries —
               the portal's parenthetical ("Гибралтар(Брит.)") and historical
               records — against what any encyclopaedia carries.
  trailing     row completion on the files whose rows end with the delimiter:
               how often the model's answer is empty, and whether the true row
               appears on any line of the answer rather than only the first,
               which is the line the library scores.

Usage:
  python src/classifier_diagnostics.py --e1 <calls_exposure_1 base> <calls_exposure_1 adapted>
                                       --e2 <calls_exposure_2 base> <calls_exposure_2 adapted>
                                       --out results/classifier_diagnostics_<stamp>.json
"""

import argparse
import csv
import json
import os
import re
import sys

from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset_registry import load_registry  # noqa: E402
from prefix_baseline import block_index, library_match, load_rows, norm_lev  # noqa: E402
from rescore_calls import feature_response, library_feature_value, query_conditions  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BINS = ((0.0, 0.2), (0.2, 0.35), (0.35, 0.5), (0.5, 1.01))
NAME_FIELD = {"mkb10_v2.csv": 3, "okved2.csv": 4, "okpdtr.csv": 3, "oksm.csv": 2}
TRAILING = ("oksm.csv", "mos_metro_stations_2022.csv", "mos_streets_omk_um_2022.csv")
PORTAL_PARENTHETICAL = re.compile(r"\([А-Яа-яЁё ]+\.?\)$")


def calls_of(path):
    return [json.loads(line) for line in open(path, encoding="utf-8")]


def model_of(calls):
    return calls[0]["model"] if calls else "?"


def path_of(dataset):
    rec = load_registry()[dataset[:-4]]
    return os.path.join(ROOT, rec["variants"]["raw"]["path"])


def name_of(line, j):
    try:
        return next(csv.reader([line]))[j].strip()
    except (csv.Error, StopIteration, IndexError):
        return ""


def gradient(calls, dataset):
    j = NAME_FIELD[dataset]
    rows = load_rows(path_of(dataset))
    index = block_index(rows)
    cells = {b: [0, 0] for b in BINS}
    for c in calls:
        if c.get("test") != "row" or c.get("kind") != "prompt" or c["dataset"] != dataset:
            continue
        located = index.get(c["prompt"].strip())
        if not located:
            continue
        truth = located[1]
        d = min(norm_lev(name_of(truth, j), name_of(r, j)) for r in c["prompt"].strip().split("\n"))
        for b in BINS:
            if b[0] <= d < b[1]:
                cells[b][1] += 1
                cells[b][0] += bool(library_match(truth, c["response"]))
    return [{"distance": f"[{a:.2f}, {b:.2f})", "matches": k, "n": n,
             "rate": round(k / n, 4) if n else None} for (a, b), (k, n) in cells.items()]


def feature_rows(calls, dataset, feature):
    """Per query: the true value (the row looked up from the conditioning
    values), the library's parsed answer, and whether they match."""
    from tabmemcheck import utils
    df = utils.load_csv_df(path_of(dataset))
    columns = [str(c) for c in df.columns]
    text = df.astype(str).apply(lambda s: s.str.strip())
    out = []
    for c in calls:
        if c.get("test") != "feature" or c["dataset"] != dataset:
            continue
        conditions = query_conditions(c["prompt"], columns)
        conditions.pop(feature, None)
        mask = None
        for name, value in conditions.items():
            hit = text[name] == value
            mask = hit if mask is None else (mask & hit)
        if mask is None or not mask.any():
            out.append(None)
            continue
        row = df.loc[mask].iloc[0]
        truth = str(row[feature]).strip()
        answer = library_feature_value(feature_response(c), feature)
        out.append({"truth": truth, "answer": answer, "match": answer is not None and truth == answer,
                    "status": str(row.get("Статус", "")),
                    "value_in_prompt": f"{feature} = {truth}" in c["prompt"].rsplit("\n\n", 1)[0]})
    return out


def feature_table(calls, baselines):
    from run_repro import FEATURES
    out = []
    for dataset, feature in FEATURES.items():
        if dataset[:-4] not in baselines:
            continue
        rows = feature_rows(calls, dataset, feature)
        if not rows:
            continue
        found = [r for r in rows if r]
        k = sum(r["match"] for r in found)
        n = len(found)
        b = baselines[dataset[:-4]]
        out.append({"dataset": dataset, "feature": feature, "matches": k, "n": n,
                    "unidentified": len(rows) - n,
                    "matches_with_value_in_few_shot": sum(r["match"] and r["value_in_prompt"] for r in found),
                    "baseline": b["baseline"], "baseline_from": b["baseline_from"],
                    "p": float(stats.binomtest(k, n, b["baseline"], alternative="greater").pvalue) if n else None})
    return out


def portal_form(calls):
    rows = [r for r in feature_rows(calls, "oksm.csv", "Полное наименование по ОКСМ") if r]
    groups = {
        "portal parenthetical": [r for r in rows if PORTAL_PARENTHETICAL.search(r["truth"])],
        "current, plain": [r for r in rows if not PORTAL_PARENTHETICAL.search(r["truth"])
                           and r["status"] == "Актуальный"],
        "historical, plain": [r for r in rows if not PORTAL_PARENTHETICAL.search(r["truth"])
                              and r["status"] == "Исторический"],
    }
    return {g: {"matches": sum(r["match"] for r in rs), "n": len(rs),
                "examples": [(r["truth"], r["answer"]) for r in rs if not r["match"]][:5]}
            for g, rs in groups.items()}


def trailing(calls, dataset):
    rows = load_rows(path_of(dataset))
    index = block_index(rows)
    n = empty = first = anywhere = 0
    for c in calls:
        if c.get("test") != "row" or c.get("kind") != "prompt" or c["dataset"] != dataset:
            continue
        located = index.get(c["prompt"].strip())
        if not located:
            continue
        n += 1
        truth = located[1].strip()
        response = c["response"]
        if not response.strip():
            empty += 1
            continue
        lines = [line.strip() for line in response.strip("\n").split("\n")]
        first += lines[0] == truth
        anywhere += truth in lines
    return {"dataset": dataset, "n": n, "empty": empty, "first_line_is_row": first,
            "row_on_any_line": anywhere}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e1", nargs=2, required=True, metavar=("BASE", "ADAPTED"))
    ap.add_argument("--e2", nargs=2, required=True, metavar=("BASE", "ADAPTED"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    baselines = json.load(open(os.path.join(ROOT, "data", "feature_baselines.json"), encoding="utf-8"))

    result = {"_what": __doc__.split("\n\n")[0], "models": {}}
    for e1, e2 in zip(args.e1, args.e2):
        c1, c2 = calls_of(e1), calls_of(e2)
        model = model_of(c1)
        entry = {
            "gradient": {d: gradient(c1, d) for d in ("mkb10_v2.csv", "okved2.csv")},
            "feature": feature_table(c2, baselines),
            "portal_form": portal_form(c2),
            "trailing": [trailing(c1 if d == "mos_metro_stations_2022.csv" else c2, d) for d in TRAILING],
        }
        result["models"][model] = entry

        print(f"\n{model}")
        print("  row completion, match rate by distance of the true name to the closest prompt name")
        for d, g in entry["gradient"].items():
            print(f"    {d:14s} " + "  ".join(f"{x['distance']} {x['matches']:>3d}/{x['n']:<3d}" for x in g))
        print("  feature completion against the conditional baseline")
        for f in entry["feature"]:
            print(f"    {f['dataset']:18s} {f['matches']:>3d}/{f['n']:<3d} baseline {f['baseline']:.4f} "
                  f"[{f['baseline_from']}]  p = {f['p']:.2e}  (value in few-shot: "
                  f"{f['matches_with_value_in_few_shot']}, unidentified {f['unidentified']})")
        print("  ОКСМ feature completion by what carries the name")
        for g, v in entry["portal_form"].items():
            print(f"    {g:22s} {v['matches']:>3d}/{v['n']}")
        print("  rows ending with the delimiter")
        for t in entry["trailing"]:
            print(f"    {t['dataset']:28s} empty {t['empty']:>3d}/{t['n']}  first line = row "
                  f"{t['first_line_is_row']}  row on any line {t['row_on_any_line']}")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print("\nwrote", args.out)


if __name__ == "__main__":
    main()
