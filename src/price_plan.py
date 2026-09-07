"""Estimate the wall-clock cost of a plan before it is queued.

A hosted session that runs out of time returns nothing, so a plan is priced
before it is run, from what the previous runs actually cost. The cost model is
fitted to the completion-mode row-completion logs of 2026-09-07 (Mistral-Nemo and
Vikhr-Nemo, nf4 on a T4): a query costs a fixed overhead plus a per-token charge
for what the model generates, with a small charge for prompt tokens. Generation
dominates, and the model stops after about one row, so the length of a row in
tokens is what makes a dataset expensive — Cyrillic rows are three to eight times
the tokens of the canon's.

Usage:
  python src/price_plan.py ru_probe ru_probe_long
  python src/price_plan.py h1b_rest --model mistralai/Mistral-Nemo-Instruct-2407
"""

import argparse
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset_registry import load_registry  # noqa: E402
from run_repro import PLANS  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# seconds; least squares over 1,394 logged row and first-token queries of
# Mistral-Nemo (nf4, one T4), median error 0.12 s. The intercept came out at
# -0.16 and is dropped.
FIXED = 0.0
PER_GENERATED_TOKEN = 0.0725
PER_PROMPT_TOKEN = 0.0037
# the model produced 1.0-2.3 rows per query on the canon before stopping; two is
# used so that the estimate errs long, since a session that runs out of time
# returns nothing
ROWS_GENERATED = 2.0
PREFIX_ROWS = 8
CHARS_PER_TOKEN = 3.0      # fallback when no tokenizer is available


def tokens_per_row(path, tokenizer):
    rows = [l for l in open(path, encoding="utf-8").read().split("\n") if l][1:]
    sample = rows[:400]
    if tokenizer is None:
        return statistics.median(len(r) for r in sample) / CHARS_PER_TOKEN
    return statistics.median(len(tokenizer(r)["input_ids"]) for r in sample)


def price(plan, tokenizer=None):
    registry = load_registry()
    lines, total = [], 0.0
    for dataset, test, queries in PLANS[plan]:
        record = registry[dataset[:-4]]
        path = os.path.join(ROOT, record["variants"]["raw"]["path"])
        row_tokens = tokens_per_row(path, tokenizer)
        if test == "header":
            seconds = queries * (FIXED + PER_GENERATED_TOKEN * 350)
        elif test == "feature":
            seconds = queries * (FIXED + PER_GENERATED_TOKEN * 16
                                 + PER_PROMPT_TOKEN * row_tokens * PREFIX_ROWS)
        else:  # row completion and first token both generate rows
            seconds = queries * (FIXED + PER_GENERATED_TOKEN * row_tokens * ROWS_GENERATED
                                 + PER_PROMPT_TOKEN * row_tokens * PREFIX_ROWS)
        total += seconds
        lines.append(f"  {dataset:28s} {test:12s} {queries:4d} queries  "
                     f"{row_tokens:5.0f} tok/row  {seconds / 60:6.0f} min")
    return lines, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("plans", nargs="+")
    ap.add_argument("--model", default="mistralai/Mistral-Nemo-Instruct-2407",
                    help="tokenizer to count with; falls back to a chars/token ratio")
    args = ap.parse_args()

    tokenizer = None
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model)
    except Exception as e:  # no cache, no network: the estimate is coarser
        print(f"tokenizer unavailable ({type(e).__name__}); using "
              f"{CHARS_PER_TOKEN} chars/token")

    for plan in args.plans:
        lines, total = price(plan, tokenizer)
        print(f"\n{plan}: {sum(q for _, _, q in PLANS[plan])} queries")
        print("\n".join(lines))
        print(f"  {'total':28s} {'':12s} {'':13s} {'':12s} {total / 3600:6.1f} h "
              f"(plus ~10 min to download and load the model)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
