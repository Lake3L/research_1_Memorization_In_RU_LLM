"""The near-duplicate half of the row-completion baseline (AMENDMENT_7).

Row completion shows the model eight consecutive rows and asks for the
ninth. On the Western canon the rows are independent observations, so the
only way to produce the ninth verbatim without having seen the file is to hit
a duplicate — the rate Bordt et al. score against. Registry exports sorted by
organisation are not like that: consecutive rows differ in a counter or a
region name, the ninth row is largely determined by the eight before it, and
the remainder is world knowledge. This module measures that and applies the
four rules of AMENDMENT_7 §2:

  R1  a query whose true row lies within tau of a prompt row is a
      near-duplicate of its own context and is not counted (tau = 0.10,
      reported at 0.05 / 0.10 / 0.20 / 0.30);
  R2  the null is max(duplicate rate, prefix-predictor rate, 3 / windows),
      never epsilon;
  R3  a positive cell needs at least one witness value — a value of a column
      named in data/witness_columns.json — that occurs nowhere in the prompt
      and is reproduced as a complete field of the scored line;
  R4  a query whose target line is not a complete record is not a row query.

Everything is computed from the call log and the frozen CSV, by the
library's own criterion (first line of the answer, substring match).

Usage:
  python src/prefix_baseline.py --group ru_pre_cutoff,fresh_control,canon        # file-level rates only
  python src/prefix_baseline.py results/calls_<base>.jsonl results/calls_<adapted>.jsonl --out results/prefix_baseline_<run>.json
  python src/prefix_baseline.py ... --file-scan     # also the file-wide near-duplicate share (slow)
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
TAU = 0.10
TAUS = (0.05, 0.10, 0.20, 0.30)
TOKENS = re.compile(r"\d+|\D+")
WITNESS_FILE = os.path.join(ROOT, "data", "witness_columns.json")


# ----------------------------------------------------------------------------
# distances, the library's criterion, the prefix-only predictor
# ----------------------------------------------------------------------------

def norm_lev(a, b):
    import jellyfish
    return jellyfish.levenshtein_distance(a, b) / max(len(a), len(b), 1)


def library_answer(response):
    """What tabmemcheck compares in completion mode: the first line of the
    response once leading and trailing newlines are removed."""
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


def predictor_hit(prefix, truth):
    """Which prefix-only predictor, if any, reproduces the true row exactly."""
    last = prefix[-1]
    if last.strip() == truth.strip():
        return "copy"
    inc = increment(prefix[-2], last) if len(prefix) >= 2 else None
    if inc is not None and inc.strip() == truth.strip():
        return "increment"
    return None


def predictor_rate(rows, prefix_rows=PREFIX_ROWS):
    """R2: the predictor's exact-hit rate over every window of the file, and
    the number of windows (for the rule-of-three floor). Fast."""
    n = hits = 0
    kinds = defaultdict(int)
    for i in range(1, len(rows) - prefix_rows):
        kind = predictor_hit(rows[i:i + prefix_rows], rows[i + prefix_rows])
        n += 1
        if kind:
            hits += 1
            kinds[kind] += 1
    return {"windows": n, "hits": hits, "rate": hits / n if n else 0.0, "by_kind": dict(kinds)}


def null_rate(duplicate_rate, predictor, windows):
    """R2: max of the duplicate rate, the predictor rate and the rule-of-three
    upper bound on a rate observed to be zero in `windows` trials."""
    floor = 3.0 / windows if windows else 0.0
    return max(duplicate_rate, predictor, floor), floor


def is_near(prefix, truth, threshold):
    lt = len(truth)
    for row in prefix:
        longest = max(lt, len(row), 1)
        if abs(lt - len(row)) / longest > threshold:
            continue
        if norm_lev(truth, row) <= threshold:
            return True
    return False


def near_duplicate_share(rows, prefix_rows=PREFIX_ROWS, threshold=TAU):
    """File-wide share of windows whose ninth row lies within `threshold` of
    one of the eight. Slow on large files; a covariate, not part of a test."""
    n = near = 0
    for i in range(1, len(rows) - prefix_rows):
        n += 1
        near += is_near(rows[i:i + prefix_rows], rows[i + prefix_rows], threshold)
    return {"windows": n, "near": near, "share": near / n if n else 0.0, "threshold": threshold}


# ----------------------------------------------------------------------------
# records, fields, witnesses
# ----------------------------------------------------------------------------

def separator_of(header):
    from metrics import infer_separator
    return infer_separator(header)


def fields_of(line, sep):
    try:
        return [x.strip() for x in next(csv.reader([line], delimiter=sep))]
    except (csv.Error, StopIteration):
        return [x.strip() for x in line.split(sep)]


def load_rows(path):
    from tabmemcheck import utils
    return utils.load_csv_rows(path)


def witnesses():
    data = json.load(open(WITNESS_FILE, encoding="utf-8"))
    return {k: v for k, v in data.items() if not k.startswith("_")}


def block_index(rows):
    index = {}
    for size in (8, 10):
        for i in range(len(rows) - size):
            index.setdefault("\n".join(rows[i:i + size]).strip(), (i, rows[i + size]))
    return index


# ----------------------------------------------------------------------------
# scoring one cell
# ----------------------------------------------------------------------------

def score_cell(calls, rows, witness_columns):
    """Every row query of one cell with what the four rules need."""
    header = rows[0]
    sep = separator_of(header)
    columns = [c.strip().lstrip("﻿") for c in fields_of(header, sep)]
    n_fields = len(columns)
    witness_idx = [columns.index(c) for c in witness_columns if c in columns]
    index = block_index(rows)
    out = []
    for call in calls:
        located = index.get(call["prompt"].strip())
        if located is None:
            continue
        i, truth = located
        prefix = call["prompt"].split("\n")
        prompt_text = call["prompt"]
        answer = library_answer(call["response"])
        answer_fields = set(fields_of(answer, sep))
        truth_fields = fields_of(truth, sep)
        record = (len(truth_fields) == n_fields) and truth[:1] not in (" ", "\t")
        rec = {
            "row": i,
            "match": library_match(truth, call["response"]),
            "predictor": predictor_hit(prefix, truth),
            "min_dist_to_prefix": round(min(norm_lev(truth, r) for r in prefix), 3),
            "is_record": record,
        }
        # R3: witness values absent from the prompt, reproduced as a field
        present = reproduced = 0
        hits = []
        if record:
            for j in witness_idx:
                v = truth_fields[j] if j < len(truth_fields) else ""
                if len(v) < 3 or v in prompt_text:
                    continue
                present += 1
                if v in answer_fields:
                    reproduced += 1
                    hits.append(columns[j])
        rec.update(witness_present=present, witness_reproduced=reproduced, witness_hits=hits)
        # the same for every other column, for the two-tier report of §7
        o_present = o_reproduced = 0
        if record:
            for j, v in enumerate(truth_fields):
                if j in witness_idx or len(v) < 3 or v in prompt_text:
                    continue
                o_present += 1
                o_reproduced += v in answer_fields
        rec.update(other_present=o_present, other_reproduced=o_reproduced)
        out.append(rec)
    return out


def summarise(scored, duplicate_rate, predictor, windows):
    n_all = len(scored)
    k_all = sum(r["match"] for r in scored)
    records = [r for r in scored if r["is_record"]]
    p0, floor = null_rate(duplicate_rate, predictor["rate"], windows)

    def test(k, n):
        return stats.binomtest(k, n, p0, alternative="greater").pvalue if n else None

    by_tau = {}
    for tau in TAUS:
        kept = [r for r in records if r["min_dist_to_prefix"] > tau]
        k, n = sum(r["match"] for r in kept), len(kept)
        by_tau[str(tau)] = {"matches": k, "n": n, "p": test(k, n)}
    main = by_tau[str(TAU)]
    witness_present = sum(r["witness_present"] for r in records)
    witness_reproduced = sum(r["witness_reproduced"] for r in records)
    positive = bool(main["p"] is not None and main["p"] < 0.05 and witness_reproduced >= 1)
    return {
        "n": n_all, "matches": k_all, "rate": k_all / n_all if n_all else None,
        "p_vs_duplicate_rate_epsilon_null": stats.binomtest(k_all, n_all, max(duplicate_rate, 1e-9), alternative="greater").pvalue if n_all else None,
        "fragment_queries": n_all - len(records),
        "near_duplicate_queries": sum(1 for r in records if r["min_dist_to_prefix"] <= TAU),
        "predictor_hits_in_cell": sum(1 for r in scored if r["predictor"]),
        "matches_that_are_predictor_hits": sum(1 for r in scored if r["match"] and r["predictor"]),
        "null": {"duplicate_rate": duplicate_rate, "predictor_rate": predictor["rate"],
                 "rule_of_three": floor, "windows": windows, "p0": p0},
        "after_rules": {"tau": TAU, **main},
        "by_tau": by_tau,
        "witness": {"present": witness_present, "reproduced": witness_reproduced,
                    "columns_hit": sorted({c for r in records for c in r["witness_hits"]})},
        "other_columns": {"present": sum(r["other_present"] for r in records),
                          "reproduced": sum(r["other_reproduced"] for r in records)},
        "positive": positive,
    }


def mcnemar(a, b, rule=True):
    """Exact McNemar on identical prompts, over the queries the rules keep."""
    def keep(r):
        return (r["is_record"] and r["min_dist_to_prefix"] > TAU) if rule else True
    va = {r["row"]: r["match"] for r in a if keep(r)}
    vb = {r["row"]: r["match"] for r in b if keep(r)}
    if set(va) != set(vb):
        return {"same_rows": False}
    both = sum(va[i] and vb[i] for i in va)
    a_only = sum(va[i] and not vb[i] for i in va)
    b_only = sum(vb[i] and not va[i] for i in va)
    p = stats.binomtest(min(a_only, b_only), a_only + b_only, 0.5).pvalue if a_only + b_only else 1.0
    return {"same_rows": True, "n": len(va), "both": both, "first_only": a_only, "second_only": b_only, "p": p}


# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("call_logs", nargs="*", help="one or two call logs; two are compared pairwise")
    ap.add_argument("--group", default="ru_pre_cutoff,fresh_control,canon")
    ap.add_argument("--file-scan", action="store_true", help="also the file-wide near-duplicate share (slow)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    registry = load_registry()
    wit = witnesses()
    groups = set(args.group.split(","))
    paths = {f"{name}.csv": os.path.join(ROOT, rec["variants"]["raw"]["path"])
             for name, rec in registry.items() if rec["group"] in groups}
    dup = {f"{name}.csv": rec["diagnostics"]["duplicate_row_share"] for name, rec in registry.items()}

    print("R2 null per file: duplicate rate, prefix-predictor rate over every window, rule-of-three floor\n")
    head = f"{'dataset':28s} {'windows':>8s} {'dup':>8s} {'predictor':>10s} {'3/W':>9s} {'p0':>9s} {'witness columns'}"
    print(head)
    print("-" * len(head))
    file_stats = {}
    for name, path in paths.items():
        if not os.path.exists(path):
            print(f"{name:28s}  (file missing — run src/fetch_data.py)")
            continue
        rows = load_rows(path)
        pred = predictor_rate(rows)
        p0, floor = null_rate(dup[name], pred["rate"], pred["windows"])
        entry = {"predictor": pred, "duplicate_rate": dup[name], "rule_of_three": floor, "p0": p0,
                 "witness_columns": wit.get(name[:-4], [])}
        if args.file_scan:
            entry["near_duplicate"] = near_duplicate_share(rows)
        file_stats[name] = entry
        print(f"{name:28s} {pred['windows']:8d} {dup[name]:8.4f} {pred['rate']:10.4f} {floor:9.2e} {p0:9.2e} "
              f"{', '.join(entry['witness_columns']) or '-'}"
              + (f"   near-dup share {entry['near_duplicate']['share']:.1%}" if args.file_scan else ""))

    result = {"files": file_stats, "cells": {}, "pairs": {}}
    scored_by_log = []
    for log in args.call_logs:
        calls = [json.loads(l) for l in open(log, encoding="utf-8")]
        model = calls[0]["model"] if calls else log
        cells = defaultdict(list)
        for c in calls:
            if c.get("test") == "row" and c.get("kind") == "prompt":
                cells[c["dataset"]].append(c)
        print(f"\n{model}  ({os.path.basename(log)})")
        head = (f"  {'dataset':26s} {'library':>8s} {'frag':>5s} {'nearQ':>6s} {'R1+R4':>8s} {'p (R2)':>9s} "
                f"{'witness':>9s} {'other':>9s}  verdict   tau-curve 0.05/0.10/0.20/0.30")
        print(head)
        print("  " + "-" * (len(head) - 2))
        scored_here = {}
        for name, group in sorted(cells.items()):
            if name not in paths or name not in file_stats:
                continue
            fs = file_stats[name]
            scored = score_cell(group, load_rows(paths[name]), fs["witness_columns"])
            scored_here[name] = scored
            s = summarise(scored, fs["duplicate_rate"], fs["predictor"], fs["predictor"]["windows"])
            result["cells"][f"{model}|{name}"] = s
            curve = "  ".join(f"{v['matches']}/{v['n']}" for v in s["by_tau"].values())
            print(f"  {name:26s} {s['matches']:>3d}/{s['n']:<4d} {s['fragment_queries']:5d} {s['near_duplicate_queries']:6d} "
                  f"{s['after_rules']['matches']:>3d}/{s['after_rules']['n']:<4d} "
                  f"{(f'{s['after_rules']['p']:.2e}' if s['after_rules']['p'] is not None else '-'):>9s} "
                  f"{s['witness']['reproduced']:>3d}/{s['witness']['present']:<5d} "
                  f"{s['other_columns']['reproduced']:>3d}/{s['other_columns']['present']:<5d}  "
                  f"{'POSITIVE' if s['positive'] else 'negative':9s} {curve}")
        scored_by_log.append((model, scored_here))

    if len(scored_by_log) == 2:
        (ma, sa), (mb, sb) = scored_by_log
        print(f"\nPaired on identical prompts, queries kept by R1 and R4: first = {ma}, second = {mb}")
        for name in sorted(set(sa) & set(sb)):
            m = mcnemar(sa[name], sb[name])
            raw = mcnemar(sa[name], sb[name], rule=False)
            result["pairs"][name] = {"after_rules": m, "all_queries": raw}
            if not m["same_rows"]:
                print(f"  {name:26s} the two logs asked different rows")
                continue
            print(f"  {name:26s} n {m['n']:3d}  both {m['both']:3d}  first only {m['first_only']:3d}  "
                  f"second only {m['second_only']:3d}  McNemar p = {m['p']:.3g}   (all queries: {raw['first_only']}/{raw['second_only']}, p = {raw['p']:.3g})")

    print("\n'library' is the count by tabmemcheck's criterion; 'frag' the queries whose target is not a record (R4);")
    print(f"'nearQ' the record queries within {TAU} of a prompt row (R1); 'R1+R4' the count over the queries kept;")
    print("'p (R2)' the one-sided exact binomial test against max(duplicate, predictor, 3/W); 'witness' the")
    print("witness values absent from the prompt and how many were reproduced as a field (R3); 'other' the same")
    print("for every non-witness column. A cell is POSITIVE only if p < 0.05 and at least one witness was reproduced.")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print("wrote", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
