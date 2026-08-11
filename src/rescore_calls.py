"""Recompute the counts of a run from its raw call log alone.

The standing rule is that every reported number regenerates from raw logs by a
committed script. This is that script for the HF runs: it reads only the JSONL
written by `hf_llm.HFLLM` and the frozen CSVs, reconstructs the ground truth from
the prompts, and counts matches again without consulting the run's own result
file. Agreement between the two is evidence that the counting is right; a
disagreement is a defect in one of them and has to be resolved before either
number is quoted.

It also produces what the live counters do not: for each cell, how close the
wrong answers were. A 7-8B model that returns a row with one digit changed has
not "failed to memorize" in the same sense as one that returns an unrelated row,
and an exact-match count cannot tell those apart.

Usage:
  python src/rescore_calls.py results/calls_*.jsonl --results results/gateA_*.json
"""

import argparse
import glob
import json
import os
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset_registry import load_registry  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_rows(path):
    from tabmemcheck import utils
    return utils.load_csv_rows(path), utils.load_csv_df(path)


def norm_lev(a, b):
    """Distance after number-format canonicalisation (AMENDMENT_3 §2)."""
    import jellyfish
    from metrics import infer_separator, normalise_numbers
    sep = infer_separator(a)
    a, b = normalise_numbers(a, sep), normalise_numbers(b, sep)
    return jellyfish.levenshtein_distance(a, b) / max(len(a), len(b), 1)


def first_line(text):
    for line in str(text).strip().split("\n"):
        if line.strip():
            return line.strip()
    return ""


def rescore_row(calls, rows):
    """The row-completion criterion: the true next row appears in the response.

    The truth is recovered from the prompt: the last user message is a block of
    consecutive rows, and the row that follows the last of them in the file is
    what the model was asked for.
    """
    index = {row: i for i, row in enumerate(rows)}
    matches, n, distances, near = 0, 0, [], 0
    for call in calls:
        prompt = [m for m in call["messages"] if m["role"] == "user"][-1]["content"]
        last = first_line(prompt.strip().split("\n")[-1])
        i = index.get(prompt.strip().split("\n")[-1].strip())
        if i is None or i + 1 >= len(rows):
            continue
        truth = rows[i + 1]
        n += 1
        if truth.strip() in call["response"].strip():
            matches += 1
        d = norm_lev(truth, first_line(call["response"]))
        distances.append(d)
        near += d <= 0.1
    return {"matches": matches, "n": n,
            "mean_normalized_levenshtein": round(sum(distances) / len(distances), 4) if distances else None,
            "near_match_rate": round(near / len(distances), 4) if distances else None,
            "unmatched_prompts": len(calls) - n}


def rescore_feature(calls, df, feature):
    """The feature-completion criterion, re-derived by looking the row up.

    The prompt states every other feature of one observation, so the row is
    identifiable and its true value for the held-out feature is known without
    trusting anything the run recorded.
    """
    matches, n, distances, near = 0, 0, [], 0
    for call in calls:
        prompt = [m for m in call["messages"] if m["role"] == "user"][-1]["content"]
        conditions = dict(re.findall(r"([A-Za-z_][\w ]*) = ([^,\n]+)", prompt))
        mask = None
        for name, value in conditions.items():
            name = name.strip()
            if name not in df.columns or name == feature:
                continue
            column = df[name].astype(str).str.strip()
            hit = column == value.strip()
            mask = hit if mask is None else (mask & hit)
        if mask is None or not mask.any():
            continue
        truth = str(df.loc[mask, feature].iloc[0]).strip()
        got = str(call["response"]).split("=")[-1].strip()
        n += 1
        matches += truth == got
        d = norm_lev(truth, got)
        distances.append(d)
        near += d <= 0.1
    return {"matches": matches, "n": n, "feature": feature,
            "mean_normalized_levenshtein": round(sum(distances) / len(distances), 4) if distances else None,
            "near_match_rate": round(near / len(distances), 4) if distances else None,
            "unidentified_rows": len(calls) - n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("call_log")
    ap.add_argument("--results", default=None, help="the run's own result file, to compare against")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from run_repro import designated_feature

    registry = load_registry()
    calls = [json.loads(l) for l in open(args.call_log, encoding="utf-8")]
    cells = defaultdict(list)
    for call in calls:
        if "messages" in call:
            cells[(call.get("test"), call.get("dataset"))].append(call)

    variant = calls[0].get("variant", "raw") if calls else "raw"
    paths = {}
    for name, rec in registry.items():
        info = rec.get("variants", {}).get(variant)
        if info:
            paths[f"{name}.csv"] = os.path.join(ROOT, info["path"])

    reported = {}
    if args.results:
        for r in json.load(open(args.results, encoding="utf-8"))["results"]:
            reported[(r.get("test"), r.get("dataset_key"))] = r

    print(f"{'test':10s} {'dataset':22s} {'rescored':>10s} {'reported':>10s} {'agree':>6s} "
          f"{'lev':>5s} {'near':>5s}  note")
    print("-" * 92)
    out = []
    for (test, dataset), group in sorted(cells.items()):
        path = paths.get(dataset)
        if path is None or test not in ("row", "feature"):
            continue
        rows, df = load_rows(path)
        if test == "row":
            got = rescore_row(group, rows)
            note = (f"{got['unmatched_prompts']} prompts not located in the file"
                    if got["unmatched_prompts"] else "")
        else:
            feature, _ = designated_feature(path)
            got = rescore_feature(group, df, feature)
            note = (f"{got['unidentified_rows']} rows not identified"
                    if got["unidentified_rows"] else "")
        ref = reported.get((test, dataset), {})
        same = (ref.get("matches") == got["matches"] and ref.get("n") == got["n"])
        lev = got["mean_normalized_levenshtein"]
        near = got["near_match_rate"]
        lev_text = f"{lev:.2f}" if lev is not None else ""
        near_text = f"{near:.0%}" if near is not None else ""
        ref_text = f"{ref.get('matches', '?')}/{ref.get('n', '?')}"
        print(f"{test:10s} {dataset:22s} {got['matches']:>4d}/{got['n']:<5d} "
              f"{ref_text:>10s} {'yes' if same else 'NO':>6s} "
              f"{lev_text:>5s} {near_text:>5s}  {note}")
        out.append({"test": test, "dataset": dataset, "rescored": got,
                    "reported_matches": ref.get("matches"), "reported_n": ref.get("n"),
                    "agrees": bool(same)})

    disagreements = [o for o in out if not o["agrees"]]
    print(f"\n{len(out) - len(disagreements)}/{len(out)} cells reproduce exactly from the raw log")
    if disagreements:
        print("DISAGREEMENT — resolve before quoting either number:",
              [(o["test"], o["dataset"]) for o in disagreements])
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print("wrote", args.out)
    return 0 if not disagreements else 1


if __name__ == "__main__":
    sys.exit(main())
