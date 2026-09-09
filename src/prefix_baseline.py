"""How much of a completed row was already in the prompt? A model-free baseline
for row completion on registry-like data, reported beside every count.

Row completion shows the model eight consecutive rows and asks for the ninth.
On the Western canon the rows are independent observations, so the only way
to produce the ninth row verbatim without having seen the file is to hit a
duplicate — which is why Bordt et al. score the test against the duplicate
rate, and why PREREGISTRATION.md §5 names the "duplicate/near-duplicate base
rate" as the bar. The Russian open-data files are not like that. A registry
export sorted by organisation carries runs of rows that differ only in a
counter or a region name (`deti.r36.nalog.ru` → `deti.r37.nalog.ru`,
`7791 Садко трейдинг …` → `7792 Садко трейдинг …`), and the ninth row is then
largely determined by the eight before it. An exact match there is pattern
continuation, not memory, and the duplicate rate does not see it.

This script measures that predictability with predictors that use only what
the model was given, in the spirit of the previous-row predictor of
AMENDMENT_6 §1:

  copy       — repeat the last prefix row;
  increment  — if the last two prefix rows differ only in digit runs, advance
               every changed run by the same step, width preserved.

Their hit rate over every eight-row window of the file is the dataset's
prefix-predictability rate: a property of the data, computable before any
model runs. Over the prompts a run actually asked it is the cell's. Beside
that, for every exact match the run produced: the distance of the true row to
the nearest prefix row, and the share of its fields that occur in no prefix
row — the content the model had to bring from outside the prompt.

Whether the predictor rate enters the decision rule is a matter for the
preregistration and its amendments, not for this script; here it is reported.

Usage:
  python src/prefix_baseline.py --group ru_pre_cutoff,fresh_control,canon
  python src/prefix_baseline.py results/calls_ru_probe_<base>.jsonl results/calls_ru_probe_<adapted>.jsonl --details
"""

import argparse
import csv
import json
import os
import re
import sys
from collections import defaultdict

from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset_registry import load_registry  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PREFIX_ROWS = 8
NEAR = 0.1
TOKENS = re.compile(r"\d+|\D+")


def norm_lev(a, b):
    import jellyfish
    return jellyfish.levenshtein_distance(a, b) / max(len(a), len(b), 1)


def library_answer(response):
    """What tabmemcheck compares against the true row in completion mode: the
    first line of the response once leading and trailing newlines are removed
    (`chat_completion.row_completion`)."""
    return response.strip("\n").split("\n")[0]


def library_match(truth, response):
    return truth.strip() in library_answer(response).strip()


def increment(prev, last):
    """Advance every digit run that changed between `prev` and `last` by the
    same step again; None when the rows differ anywhere else."""
    tp, tl = TOKENS.findall(prev), TOKENS.findall(last)
    if len(tp) != len(tl):
        return None
    out, changed = [], False
    for a, b in zip(tp, tl):
        if a == b:
            out.append(b)
        elif a.isdigit() and b.isdigit():
            nxt = int(b) + (int(b) - int(a))
            if nxt < 0:
                return None
            out.append(str(nxt).zfill(len(b)))
            changed = True
        else:
            return None
    return "".join(out) if changed else None


def predictor_hits(prefix, truth):
    """Does either prefix-only predictor reproduce the true row exactly?"""
    last = prefix[-1]
    if last.strip() == truth.strip():
        return "copy"
    inc = increment(prefix[-2], last) if len(prefix) >= 2 else None
    if inc is not None and inc.strip() == truth.strip():
        return "increment"
    return None


def fields(line):
    from metrics import infer_separator
    sep = infer_separator(line)
    try:
        parsed = next(csv.reader([line], delimiter=sep))
    except (csv.Error, StopIteration):
        parsed = line.split(sep)
    return [f.strip() for f in parsed if f.strip()]


def novel_fields(truth, prefix):
    seen = set()
    for row in prefix:
        seen.update(fields(row))
    own = fields(truth)
    return [f for f in own if f not in seen], own


IP_LIKE = re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")
NUMBER = re.compile(r"^\d+([.,]\d+)?$")
CYRILLIC = re.compile("[А-Яа-яЁё]")


def field_kind(value):
    """A coarse split of a field the model had to bring from outside the prompt:
    arbitrary identifiers (IP addresses, numbers) that only the file could
    supply, against domains and Cyrillic names that world knowledge or a
    naming pattern can."""
    if IP_LIKE.match(value):
        return "ip"
    if NUMBER.match(value):
        return "number"
    if CYRILLIC.search(value):
        return "cyrillic text"
    if "." in value and " " not in value:
        return "domain"
    return "other"


def novel_recall(truth, prefix, answer):
    """For every field of the true row that occurs in no prefix row: its kind,
    and whether the model's answer contains it verbatim. Counted over all
    queries, not only exact matches — the §7 two-tier view of one cell."""
    novel, _ = novel_fields(truth, prefix)
    answer_fields = set(fields(answer))
    return [(field_kind(f), (f in answer_fields) or (f in answer)) for f in novel]


def load_rows(path):
    from tabmemcheck import utils
    return utils.load_csv_rows(path)


def near_prefix(prefix, truth):
    return min(norm_lev(truth, r) for r in prefix)


def is_near(prefix, truth, threshold=NEAR):
    """Whether the true row lies within `threshold` of some prefix row. A length
    difference alone bounds the normalised distance from below, so most pairs
    are settled without computing an edit distance; the file-wide scan over
    32,000 windows of adult-train needs that."""
    lt = len(truth)
    for row in prefix:
        longest = max(lt, len(row), 1)
        if abs(lt - len(row)) / longest > threshold:
            continue
        if norm_lev(truth, row) <= threshold:
            return True
    return False


def dataset_rate(rows):
    """Prefix-predictability over every eight-row window of the file: how often
    a prefix-only predictor reproduces the ninth row, and how often the ninth
    row lies within NEAR of one of the eight (a near-duplicate of its context)."""
    n = hits = near = 0
    kinds = defaultdict(int)
    for i in range(1, len(rows) - PREFIX_ROWS):
        prefix, truth = rows[i:i + PREFIX_ROWS], rows[i + PREFIX_ROWS]
        kind = predictor_hits(prefix, truth)
        n += 1
        if kind:
            hits += 1
            kinds[kind] += 1
        near += is_near(prefix, truth)
    return {"windows": n, "hits": hits, "rate": hits / n if n else None, "by_kind": dict(kinds),
            "near_duplicate_windows": near, "near_duplicate_share": near / n if n else None}


def block_index(rows):
    index = {}
    for size in (8, 10):
        for i in range(len(rows) - size):
            index.setdefault("\n".join(rows[i:i + size]).strip(), (i, rows[i + size]))
    return index


def score_cell(calls, rows):
    """Every row query of one cell, scored by the library's own criterion, with
    the prefix-only predictor and the novelty of each exact match beside it."""
    index = block_index(rows)
    out = []
    for call in calls:
        located = index.get(call["prompt"].strip())
        if located is None:
            continue
        i, truth = located
        prefix = call["prompt"].split("\n")
        match = library_match(truth, call["response"])
        kind = predictor_hits(prefix, truth)
        dmin = round(near_prefix(prefix, truth), 3)
        rec = {"row": i, "match": match, "predictor": kind, "min_dist_to_prefix": dmin,
               "near_duplicate_query": dmin <= NEAR,
               "novel_recall": novel_recall(truth, prefix, library_answer(call["response"]))}
        if match:
            novel, own = novel_fields(truth, prefix)
            rec.update(dist_to_prev=round(norm_lev(truth, prefix[-1]), 3),
                       novel_fields=novel, novel_share=round(len(novel) / len(own), 3) if own else None,
                       truth=truth)
        out.append(rec)
    return out


def summarise(scored, duplicate_baseline, predictor_rate):
    n = len(scored)
    matches = [r for r in scored if r["match"]]
    k = len(matches)
    pred_in_cell = sum(1 for r in scored if r["predictor"])
    both = sum(1 for r in matches if r["predictor"])
    near_queries = sum(1 for r in scored if r["near_duplicate_query"])
    near = sum(1 for r in matches if r["near_duplicate_query"])
    novel_shares = [r["novel_share"] for r in matches if r["novel_share"] is not None]
    fully_novel = sum(1 for r in matches if r["novel_share"] and r["novel_share"] >= 0.5)
    stronger = max(duplicate_baseline, predictor_rate or 0.0)
    n_far = n - near_queries
    recall = defaultdict(lambda: [0, 0])
    for r in scored:
        for kind, hit in r["novel_recall"]:
            recall[kind][0] += 1
            recall[kind][1] += hit
    return {
        "novel_field_recall": {k: {"present": v[0], "recalled": v[1]} for k, v in sorted(recall.items())},
        "n": n, "matches": k, "rate": k / n if n else None,
        "predictor_hits_in_cell": pred_in_cell,
        "matches_that_are_predictor_hits": both,
        "near_duplicate_queries": near_queries,
        "matches_within_near_of_a_prefix_row": near,
        "matches_with_novel_share_at_least_half": fully_novel,
        "mean_novel_share_of_matches": round(sum(novel_shares) / len(novel_shares), 3) if novel_shares else None,
        "p_vs_duplicate": stats.binomtest(k, n, max(duplicate_baseline, 1e-9), alternative="greater").pvalue if n else None,
        "p_vs_duplicate_or_predictor": stats.binomtest(k, n, max(stronger, 1e-9), alternative="greater").pvalue if n else None,
        "p_excluding_predictor_hits": (stats.binomtest(k - both, n - pred_in_cell, max(duplicate_baseline, 1e-9),
                                                       alternative="greater").pvalue if n - pred_in_cell > 0 else None),
        "matches_excluding_near_duplicate_queries": k - near,
        "n_excluding_near_duplicate_queries": n_far,
        "p_excluding_near_duplicate_queries": (stats.binomtest(k - near, n_far, max(duplicate_baseline, 1e-9),
                                                               alternative="greater").pvalue if n_far > 0 else None),
    }


def mcnemar(a, b):
    """Exact McNemar on identical prompts: rows the first log matched and the
    second did not, and the reverse."""
    va = {r["row"]: r["match"] for r in a}
    vb = {r["row"]: r["match"] for r in b}
    if set(va) != set(vb):
        return {"same_rows": False}
    both = sum(va[i] and vb[i] for i in va)
    a_only = sum(va[i] and not vb[i] for i in va)
    b_only = sum(vb[i] and not va[i] for i in va)
    p = stats.binomtest(min(a_only, b_only), a_only + b_only, 0.5).pvalue if a_only + b_only else 1.0
    return {"same_rows": True, "both": both, "first_only": a_only, "second_only": b_only, "p": p}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("call_logs", nargs="*", help="one or two call logs; two are compared pairwise")
    ap.add_argument("--group", default="ru_pre_cutoff,fresh_control,canon")
    ap.add_argument("--details", action="store_true", help="print every exact match")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    registry = load_registry()
    groups = set(args.group.split(","))
    paths = {f"{name}.csv": os.path.join(ROOT, rec["variants"]["raw"]["path"])
             for name, rec in registry.items() if rec["group"] in groups}
    baselines = {f"{name}.csv": rec["diagnostics"]["duplicate_row_share"] for name, rec in registry.items()}

    print(f"Prefix-only predictors (copy, increment) and near-duplicate share (ninth row within "
          f"{NEAR} of one of the eight) over every {PREFIX_ROWS}-row window\n")
    header = (f"{'dataset':28s} {'windows':>8s} {'pred':>5s} {'pred rate':>9s} {'near-dup':>8s} "
              f"{'near share':>10s} {'dup base':>9s}  kinds")
    print(header)
    print("-" * len(header))
    data_rates = {}
    for name, path in paths.items():
        if not os.path.exists(path):
            print(f"{name:28s}  (file missing — run src/fetch_data.py)")
            continue
        rows = load_rows(path)
        r = dataset_rate(rows)
        data_rates[name] = r
        print(f"{name:28s} {r['windows']:8d} {r['hits']:5d} {r['rate']:9.3%} {r['near_duplicate_windows']:8d} "
              f"{r['near_duplicate_share']:10.3%} {baselines[name]:9.4f}  {r['by_kind']}")

    result = {"dataset_rates": data_rates, "cells": {}, "pairs": {}}
    scored_by_log = []
    for log in args.call_logs:
        calls = [json.loads(l) for l in open(log, encoding="utf-8")]
        model = calls[0]["model"] if calls else log
        cells = defaultdict(list)
        for c in calls:
            if c.get("test") == "row" and c.get("kind") == "prompt":
                cells[c["dataset"]].append(c)
        print(f"\n{model}  ({os.path.basename(log)})")
        head = (f"  {'dataset':28s} {'match':>8s} {'pred':>5s} {'m∩p':>4s} {'nearQ':>5s} {'m∩near':>6s} "
                f"{'novel':>6s} {'p dup':>9s} {'p dup|pred':>10s} {'far match':>9s} {'p far':>9s}")
        print(head)
        print("  " + "-" * (len(head) - 2))
        scored_here = {}
        for name, group in sorted(cells.items()):
            if name not in paths:
                continue
            scored = score_cell(group, load_rows(paths[name]))
            scored_here[name] = scored
            s = summarise(scored, baselines[name], data_rates.get(name, {}).get("rate"))
            result["cells"][f"{model}|{name}"] = s
            far = f"{s['matches_excluding_near_duplicate_queries']}/{s['n_excluding_near_duplicate_queries']}"
            print(f"  {name:28s} {s['matches']:>4d}/{s['n']:<3d} {s['predictor_hits_in_cell']:5d} "
                  f"{s['matches_that_are_predictor_hits']:4d} {s['near_duplicate_queries']:5d} "
                  f"{s['matches_within_near_of_a_prefix_row']:6d} "
                  f"{(f'{s['mean_novel_share_of_matches']:.2f}' if s['mean_novel_share_of_matches'] is not None else '-'):>6s} "
                  f"{s['p_vs_duplicate']:9.2e} {s['p_vs_duplicate_or_predictor']:10.2e} {far:>9s} "
                  f"{(f'{s['p_excluding_near_duplicate_queries']:.2e}' if s['p_excluding_near_duplicate_queries'] is not None else '-'):>9s}")
            if s["matches"]:
                recall = ", ".join(f"{k} {v['recalled']}/{v['present']}"
                                   for k, v in s["novel_field_recall"].items())
                print(f"      fields absent from the prompt, recalled verbatim over all {s['n']} queries: {recall}")
            if args.details:
                for r in scored:
                    if r["match"]:
                        print(f"      row {r['row']:5d}  pred={r['predictor'] or '-':9s} dmin={r['min_dist_to_prefix']:.3f} "
                              f"novel={r['novel_share']}  {r['novel_fields'][:3]}")
        scored_by_log.append((model, scored_here))

    if len(scored_by_log) == 2:
        (ma, sa), (mb, sb) = scored_by_log
        print(f"\nPaired on identical prompts: first = {ma}, second = {mb}")
        for name in sorted(set(sa) & set(sb)):
            m = mcnemar(sa[name], sb[name])
            result["pairs"][name] = m
            if not m["same_rows"]:
                print(f"  {name:28s} the two logs asked different rows")
                continue
            print(f"  {name:28s} both {m['both']:3d}  first only {m['first_only']:3d}  "
                  f"second only {m['second_only']:3d}  McNemar p = {m['p']:.3g}")

    print("\nRead 'pred' as the number of asked rows a prefix-only predictor reproduces exactly and")
    print(f"'m∩p' as the exact matches among them; 'nearQ' as the asked rows that lie within {NEAR}")
    print("of a prefix row (near-duplicates of their own context) and 'm∩near' as the exact matches")
    print("among those; 'novel' as the mean share of a matched row's fields that occur in no prefix")
    print("row. The p-values are one-sided exact binomial tests against the duplicate rate, against")
    print("the larger of duplicate and predictor rate, and — 'far' — over the queries that are not")
    print("near-duplicates of their context, against the duplicate rate.")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print("wrote", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
