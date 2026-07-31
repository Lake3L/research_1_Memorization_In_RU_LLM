"""Reproduce the memorization tests of Bordt et al. (COLM 2024) — positive-control
gate of PREREGISTRATION.md §8.

Runs unmodified tabmemcheck tests against the model versions used in the paper
and compares the counts to Tables 2/5/6. Supports --dry-run to price a plan
before spending anything.

Usage:
  python run_repro.py --plan gate --model gpt-3.5-turbo-16k --dry-run
  python run_repro.py --plan gate --model gpt-3.5-turbo-16k --budget 2.0
"""

import argparse
import io
import json
import os
import re
import sys
from contextlib import redirect_stdout
from datetime import datetime, timezone

import numpy as np
from scipy import stats

import tabmemcheck as tabmem
from tabmemcheck import utils

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from budget_llm import BudgetedOpenAILLM, BudgetExceeded  # noqa: E402
from mock_llm import PerfectMemorizer, format_echo_mock  # noqa: E402

# max_tokens per test: the tests need one row (or one feature value) at most,
# and the cap arithmetic is worst-case, so a tight limit buys budget headroom
MAX_TOKENS = {"header": 300, "row": 100, "feature": 60, "first_token": 100}

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Reference values from the paper. Row/feature completion: Table 5 (p. 21).
# First token: Table 6 (p. 21). Header: Table 2 (p. 4), pass/fail.
PAPER = {
    ("iris.csv", "row"): {"gpt-3.5": (35, 136), "gpt-4": (125, 136)},
    ("uci-wine.csv", "row"): {"gpt-3.5": (16, 164), "gpt-4": (84, 164)},
    ("openml-diabetes.csv", "row"): {"gpt-3.5": (18, 250), "gpt-4": (79, 250)},
    ("adult-train.csv", "row"): {"gpt-3.5": (0, 250), "gpt-4": (0, 250)},
    ("california-housing.csv", "row"): {"gpt-3.5": (0, 250), "gpt-4": (0, 250)},
    ("uci-wine.csv", "feature"): {"gpt-3.5": (77, 178), "gpt-4": (131, 178)},
    ("openml-diabetes.csv", "feature"): {"gpt-3.5": (237, 250), "gpt-4": (243, 250)},
    ("adult-train.csv", "feature"): {"gpt-3.5": (0, 250), "gpt-4": (0, 250)},
    ("california-housing.csv", "feature"): {"gpt-3.5": (0, 250), "gpt-4": (1, 250)},
    ("iris.csv", "first_token"): {"gpt-3.5": (88, 136), "gpt-4": (131, 136), "baseline": 0.50},
    ("openml-diabetes.csv", "first_token"): {"gpt-3.5": (42, 250), "gpt-4": (95, 250), "baseline": 0.25},
    ("adult-train.csv", "first_token"): {"gpt-3.5": (59, 250), "gpt-4": (68, 250), "baseline": 0.26},
    ("iris.csv", "header"): {"gpt-3.5": "pass", "gpt-4": "pass"},
    ("uci-wine.csv", "header"): {"gpt-3.5": "pass", "gpt-4": "pass"},
    ("openml-diabetes.csv", "header"): {"gpt-3.5": "pass", "gpt-4": "pass"},
    ("adult-train.csv", "header"): {"gpt-3.5": "pass", "gpt-4": "pass"},
    ("california-housing.csv", "header"): {"gpt-3.5": "pass", "gpt-4": "pass"},
}

# The paper's feature choices (Table 5 caption).
FEATURES = {
    "uci-wine.csv": "malic_acid",
    "openml-diabetes.csv": "DiabetesPedigreeFunction",
    "adult-train.csv": "fnlwgt",
    "california-housing.csv": "median_income",
}

CANON = ["iris.csv", "uci-wine.csv", "openml-diabetes.csv",
         "adult-train.csv", "california-housing.csv"]

PLANS = {
    # full paper-scale protocol; priced by --dry-run before use
    "full": [(d, t, 250) for d in CANON for t in ("header", "row", "feature", "first_token")],
    # positive-control gate scaled to a small budget: cheap tests everywhere,
    # expensive row completion only where the paper reports a strong signal
    # (iris) and one negative control (adult)
    "gate": [
        ("iris.csv", "header", 4), ("uci-wine.csv", "header", 4),
        ("openml-diabetes.csv", "header", 4), ("adult-train.csv", "header", 4),
        ("california-housing.csv", "header", 4),
        ("openml-diabetes.csv", "feature", 50), ("uci-wine.csv", "feature", 50),
        ("adult-train.csv", "feature", 50), ("california-housing.csv", "feature", 50),
        ("iris.csv", "row", 50), ("openml-diabetes.csv", "row", 25),
        ("adult-train.csv", "row", 25),
        ("iris.csv", "first_token", 50),
    ],
    "header_only": [(d, "header", 4) for d in CANON],
    # gpt-4-0613 costs 60x gpt-3.5-turbo-0125 per token, so buy only the
    # sharpest contrasts: iris row completion (paper: 92% vs GPT-3.5's 26%),
    # diabetes feature completion (97%), and one negative control
    "gate4": [
        ("iris.csv", "row", 25),
        ("openml-diabetes.csv", "feature", 25),
        ("adult-train.csv", "feature", 25),
    ],
}


ANSI = re.compile(r"\x1b\[[0-9;]*m")


def duplicate_rate(csv_file):
    rows = utils.load_csv_rows(csv_file)
    return 1 - len(set(rows)) / len(rows)


def run_one(llm, csv_file, test, num_queries, seed):
    """Run one test, returning a result dict. Output of tabmemcheck is captured."""
    rng = np.random.default_rng(seed)
    buf = io.StringIO()
    result = {"dataset": csv_file, "test": test, "requested_queries": num_queries}
    tabmem.config.max_tokens = MAX_TOKENS.get(test, 300)

    with redirect_stdout(buf):
        if test == "header":
            prompt, completion, response = tabmem.header_test(csv_file, llm, rng=rng)
            match = 0
            for a, b in zip(completion, response):
                if a != b:
                    break
                match += 1
            # Bordt's criterion: the model completes at least the next full row.
            # The split falls mid-row, so that means matching the remainder of
            # the current row, the newline, and the whole row after it.
            result["completion_prefix_match_chars"] = match
            segments = completion.split("\n")
            needed = len(segments[0]) + 1 + (len(segments[1]) if len(segments) > 1 else 10**9)
            result["chars_needed_for_next_row"] = needed
            result["verdict"] = "pass" if match >= needed else "fail"
            result["response_head"] = response[:200]
            result["true_head"] = completion[:200]

        elif test == "row":
            suffixes, responses = tabmem.row_completion_test(
                csv_file, llm, num_queries=num_queries, rng=rng, print_levenshtein=False
            )
            n = len(responses)
            k = sum(1 for s, r in zip(suffixes, responses) if s.strip() in r.strip())
            base = duplicate_rate(csv_file)
            result.update(matches=k, n=n, rate=k / n if n else 0.0,
                          baseline_rate=base,
                          p_value=float(stats.binomtest(k, n, max(base, 1e-9),
                                                        alternative="greater").pvalue) if n else None)

        elif test == "feature":
            feature = FEATURES.get(csv_file)
            values, responses = tabmem.feature_completion_test(
                csv_file, llm, feature_name=feature, num_queries=num_queries, rng=rng
            )
            n = len(responses)
            k = sum(1 for v, r in zip(values, responses)
                    if str(v).strip() == str(r).strip())
            df = utils.load_csv_df(csv_file)
            mode_rate = float(df[feature].astype(str).value_counts(normalize=True).iloc[0])
            result.update(feature=feature, matches=k, n=n, rate=k / n if n else 0.0,
                          mode_baseline=mode_rate,
                          p_value=float(stats.binomtest(k, n, max(mode_rate, 1e-9),
                                                        alternative="greater").pvalue) if n else None)

        elif test == "first_token":
            tabmem.first_token_test(csv_file, llm, num_queries=num_queries, rng=rng)
            # ANSI colour codes carry digits that would confuse the parsing below
            out = ANSI.sub("", buf.getvalue())
            m = re.search(r"First Token Test: \D*(\d+)/(\d+)", out)
            b = re.search(r"most common first token\)\D*(\d+)/(\d+)", out)
            if m:
                k, n = int(m.group(1)), int(m.group(2))
                result.update(matches=k, n=n, rate=k / n)
            if b:
                bk, bn = int(b.group(1)), int(b.group(2))
                result.update(baseline_matches=bk, baseline_rate=bk / bn)
                if m:
                    result["p_value"] = float(stats.binomtest(
                        int(m.group(1)), int(m.group(2)), max(bk / bn, 1e-9),
                        alternative="greater").pvalue)
            if "aborted" in out.lower() or "reject" in out.lower():
                result["note"] = "row-independence pre-check flagged; see stdout"
        else:
            raise ValueError(test)

    result["stdout"] = ANSI.sub("", buf.getvalue())[-2000:]
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default="gate", choices=list(PLANS))
    ap.add_argument("--model", default="gpt-3.5-turbo-16k")
    ap.add_argument("--budget", type=float, default=1.0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--mock", default="none", choices=["none", "perfect", "echo"],
                    help="dry-run stand-in model: 'perfect' must score ~100%, "
                         "'echo' must score ~0% — validates the counting code")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--only", default=None, help="comma-separated test names to keep")
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "results"))
    args = ap.parse_args()

    if not args.dry_run and not os.environ.get("OPENAI_API_KEY"):
        key_file = os.path.join(ROOT, "api-secretkey-openai.txt")
        if os.path.exists(key_file):
            os.environ["OPENAI_API_KEY"] = open(key_file, encoding="utf-8").read().strip()
        else:
            sys.exit("no OPENAI_API_KEY and no key file")

    os.makedirs(args.out_dir, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    tag = f"{args.plan}_{args.model}_{'dry' if args.dry_run else 'live'}_{stamp}"
    log_path = os.path.join(args.out_dir, f"chatlog_{tag}.jsonl")

    tabmem.config.print_prompts = False
    tabmem.config.print_responses = False

    llm = BudgetedOpenAILLM(model=args.model, budget_usd=args.budget,
                            dry_run=args.dry_run, log_path=log_path)

    plan = PLANS[args.plan]
    if args.only:
        keep = set(args.only.split(","))
        plan = [p for p in plan if p[1] in keep]

    results, prev_cost = [], 0.0
    for csv_file, test, nq in plan:
        if args.dry_run and args.mock != "none":
            llm.mock_fn = (PerfectMemorizer(csv_file) if args.mock == "perfect"
                           else format_echo_mock)
        try:
            r = run_one(llm, csv_file, test, nq, args.seed)
        except BudgetExceeded as e:
            print(f"[BUDGET STOP] {csv_file} {test}: {e}")
            results.append({"dataset": csv_file, "test": test, "error": "budget_exceeded"})
            break
        except Exception as e:  # keep going: one broken cell must not kill the run
            print(f"[ERROR] {csv_file} {test}: {type(e).__name__}: {e}")
            results.append({"dataset": csv_file, "test": test, "error": f"{type(e).__name__}: {e}"})
            continue
        r["cost_usd"] = round(llm.cost_usd - prev_cost, 4)
        r["calls"] = llm.n_calls
        prev_cost = llm.cost_usd
        results.append(r)
        ref = PAPER.get((csv_file, test), {})
        summary = (f"{r.get('matches')}/{r.get('n')}" if "matches" in r
                   else r.get("verdict", "?"))
        print(f"{csv_file:26s} {test:12s} -> {str(summary):10s} "
              f"paper: {ref} (${r['cost_usd']:.3f})")

    out = {
        "run": tag,
        "timestamp_utc": stamp,
        "model_requested": args.model,
        "plan": args.plan,
        "seed": args.seed,
        "tabmemcheck_version": getattr(tabmem, "__version__", "unknown"),
        "budget_usd": args.budget,
        "llm_summary": llm.summary(),
        "paper_reference": {f"{k[0]}|{k[1]}": v for k, v in PAPER.items()},
        "results": results,
    }
    out_path = os.path.join(args.out_dir, f"repro_{tag}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n{llm.summary()}\nwrote {out_path}")


if __name__ == "__main__":
    main()
