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


def block_index(rows, prefix_sizes=(8, 10)):
    """Map a whole prefix block to the row that follows it.

    Matching on the last row of the prompt alone is wrong wherever a dataset
    contains duplicate rows: the lookup lands on the first occurrence and the row
    taken as ground truth is then a different row. Iris carries 2% duplicates,
    which is enough to move a count. Blocks of eight or ten rows are unique in
    practice, so the reconstruction is matched on the whole block.
    """
    index = {}
    for size in prefix_sizes:
        for i in range(len(rows) - size):
            index.setdefault(chr(10).join(rows[i:i + size]).strip(), rows[i + size])
    return index


def prompt_text(call):
    """The user-visible prompt, whichever prompting mode produced the call."""
    if call.get("kind") == "prompt":
        return call["prompt"]
    return [m for m in call["messages"] if m["role"] == "user"][-1]["content"]


def first_line(text):
    for line in str(text).strip().split("\n"):
        if line.strip():
            return line.strip()
    return ""


def library_answer(call):
    """The part of the response tabmemcheck actually compares.

    In completion mode `chat_completion.row_completion` keeps only the first
    line of the response after leading and trailing newlines are removed; the
    chat path compares the whole response. Scoring the whole response in both
    modes over-counted on govdomains, where the model sometimes repeats the last
    prefix row first and produces the true row on its second line: 38 against
    the library's 32. The criterion is the library's, so the re-score follows it.
    """
    response = call["response"]
    if call.get("kind") == "prompt":
        return response.strip("\n").split("\n")[0]
    return response


def rescore_row(calls, rows):
    """The row-completion criterion: the true next row appears in the response.

    The truth is recovered from the prompt: the last user message is a block of
    consecutive rows, and the row that follows the last of them in the file is
    what the model was asked for.
    """
    index = block_index(rows)
    matches, n, distances, near = 0, 0, [], 0
    for call in calls:
        truth = index.get(prompt_text(call).strip())
        if truth is None:
            continue
        n += 1
        if truth.strip() in library_answer(call).strip():
            matches += 1
        d = norm_lev(truth, first_line(call["response"]))
        distances.append(d)
        near += d <= 0.1
    return {"matches": matches, "n": n,
            "mean_normalized_levenshtein": round(sum(distances) / len(distances), 4) if distances else None,
            "near_match_rate": round(near / len(distances), 4) if distances else None,
            "unmatched_prompts": len(calls) - n}


def feature_response(call):
    """The text the feature-completion test actually receives.

    In completion mode the library wraps the model in
    `ChatWrappedLLM(..., ends_with="\\n\\n")` and cuts every answer at its first
    blank line (`functions.py`, feature_completion_test). The call log is
    written below that wrapper and holds the generation before the cut, so an
    answer that *starts* with a blank line — `"\\n\\nНаименование = …"` — is
    logged in full but reaches the test as the empty string. Chat-mode calls
    are not wrapped and are returned unchanged.
    """
    response = str(call["response"])
    if call.get("kind") == "prompt" and "\n\n" in response:
        response = response[:response.find("\n\n")]
    return response


def library_feature_value(response, feature):
    """The value tabmemcheck reads out of a feature-completion answer.

    `utils.parse_feature_string` looks for the magic string `"<feature> = "`,
    takes what follows it up to the next occurrence of a magic string (then up
    to the last comma before it) or, failing that, up to the final delimiter,
    a newline. Taking the text after the *last* `=` in the response instead
    missed answers the model went on to elaborate — `malic_acid = 1.75\\n\\nThe
    task is …` — and re-scored two block B feature cells at 0 where the library
    counted 1. None when the answer never names the feature.
    """
    # The library walks the whole answer in a loop and overwrites the value at
    # every occurrence of the magic string, so what it scores is the value after
    # the *last* one. Taking the first instead agreed with the library as long as
    # a model stopped after one answer; Vikhr-Nemo goes on to write the next
    # few-shot block, and the first-occurrence reading over-counted its
    # feature cells by 3 on okved2 and 7 on oksm (session E2). This is a port of
    # `utils.parse_feature_string` for one feature with `final_delimiter="\n"`,
    # expression for expression, including its slice to `rfind(",")` = -1.
    magic = feature + " = "
    s, value = response, None
    while len(s) > 3:
        start = s.find(magic)
        if start == -1:
            break
        following = s.find(magic, start + 3)
        if following != -1:
            end = following
            value = s[start + len(magic):s[:end].rfind(",")].strip()
        else:
            newline = s[start + len(magic):].find("\n")
            end = start + len(magic) + newline if newline > -1 else len(s)
            value = s[start + len(magic):end].strip()
        s = s[end:]
    return value


def query_conditions(prompt, columns):
    """The conditioning values of the observation the query asks about.

    The library writes an observation as `name = value, name = value, …`.
    Column names can be Cyrillic or contain spaces and parentheses, and values
    can themselves contain `, ` (ОКВЭД names, "МОЛДОВА, РЕСПУБЛИКА"), so the
    last block of the prompt is cut at `<known column> = ` boundaries — a known
    name at the start of the block or right after `, ` — rather than matched by
    a pattern over names. The pattern this replaces accepted Latin names only
    and silently identified no row on a Russian file.
    """
    block = prompt.strip().split("\n\n")[-1]
    bounds = []
    for name in columns:
        key = f"{name} = "
        start = 0
        while True:
            i = block.find(key, start)
            if i < 0:
                break
            if i == 0 or block[i - 2:i] == ", ":
                bounds.append((i, name))
            start = i + 1
    bounds.sort()
    conditions = {}
    for k, (i, name) in enumerate(bounds):
        end = bounds[k + 1][0] - 2 if k + 1 < len(bounds) else len(block)
        conditions[name] = block[i + len(name) + 3:end].strip().rstrip(",").strip()
    return conditions


def rescore_feature(calls, df, feature):
    """The feature-completion criterion, re-derived by looking the row up.

    The prompt states every other feature of one observation, so the row is
    identifiable and its true value for the held-out feature is known without
    trusting anything the run recorded.
    """
    matches, n, distances, near = 0, 0, [], 0
    columns = [str(c) for c in df.columns]
    for call in calls:
        prompt = prompt_text(call)
        conditions = query_conditions(prompt, columns)
        mask = None
        for name, value in conditions.items():
            if name == feature:
                continue
            column = df[name].astype(str).str.strip()
            hit = column == value.strip()
            mask = hit if mask is None else (mask & hit)
        if mask is None or not mask.any():
            continue
        truth = str(df.loc[mask, feature].iloc[0]).strip()
        got = library_feature_value(feature_response(call), feature)
        n += 1
        matches += truth == got
        got = got or ""
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
        if "messages" in call or "prompt" in call:
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
