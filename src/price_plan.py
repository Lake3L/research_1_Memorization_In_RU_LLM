"""Price a plan before it is queued: an upper bound on its wall-clock time.

A hosted session that runs out of time returns nothing, so every plan is
priced before it runs. The figure used for planning is an **upper bound**: each
answer is assumed to run to the token budget the library gives it, which is
what a model without a stopping condition does under completion prompting.

What the cost depends on, measured (2026-10-04) on every logged completion-mode
call of block C — 10,719 calls of Mistral-Nemo, 9,166 of Vikhr-Nemo and 12,804
of the YandexGPT pair, nf4 on one T4 — by least squares on a sample of 2,500
per family (median absolute error 0.10–0.11 s, totals reproduced to 0.1%):

  seconds = fixed + per_token * out + per_token_per_1k_context * out * (in + out/2) / 1000
                  + per_1k_prompt * in / 1000

`in` is the prompt in tokens and `out` the tokens generated. The budget `out`
is set by the library, not by us:

  row, first token   1 + the length of the true next row **in characters**
                     (tabmemcheck send_completion(..., max_tokens=1 + len(suffix));
                     the logs show the same budget on first-token calls). A
                     Cyrillic row of 1,300 characters is a budget of 1,300
                     tokens — about six rows.
  header             1,000 tokens (MAX_TOKENS_REFERENCE)
  feature            64 tokens

The previous version of this script (2026-09-07 to 2026-10-03) charged two rows
of generation per query in tokens. That held where a row is a few dozen
characters and was wrong by a factor of 3.3 on russian_retail (session C2:
YandexGPT 9.7 h against 3.0 h estimated).

The budget-bound figure is a fit, not a ceiling: runs scatter around it by a
few percent (a slower card, a busier host). The planning figure is therefore
the budget bound times SAFETY, and `--validate` replays every completed block
B and C run against it; no run may exceed it, and SAFETY is raised, never the
plan squeezed, if one does.

Usage:
  python src/price_plan.py ru_probe exposure_1 --model yandex/YandexGPT-5-Lite-8B-pretrain
  python src/price_plan.py --validate
"""

import argparse
import glob
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset_registry import load_registry  # noqa: E402
from run_repro import PLANS  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# fitted 2026-10-04 (see the module docstring); seconds
COST = {
    "nemo-12b": {"fixed": 0.30, "per_token": 0.0715, "per_token_per_1k_context": 0.0244,
                 "per_1k_prompt": 1.62, "measured_on": "Mistral-Nemo-Instruct-2407, Vikhr-Nemo-12B"},
    "yandex-8b": {"fixed": 0.37, "per_token": 0.0516, "per_token_per_1k_context": 0.0169,
                  "per_1k_prompt": 0.99, "measured_on": "YandexGPT-5-Lite-8B pretrain and instruct"},
    # fitted 2026-10-05 on session O1 (912 and 2,748 calls). Their prompts were
    # canon prompts of 100-900 tokens, too short to identify how the cost per
    # token grows with the context, so that term is the 8B family's, the
    # nearest measured on long Russian prompts, rather than the much smaller
    # value these short prompts gave (0.0015 and 0.0032)
    "qwen-7b": {"fixed": 0.07, "per_token": 0.0594, "per_token_per_1k_context": 0.0169,
                "per_1k_prompt": 0.85, "measured_on": "Qwen2.5-7B-Instruct, canon only"},
    "olmo-7b": {"fixed": 0.0, "per_token": 0.0449, "per_token_per_1k_context": 0.0169,
                "per_1k_prompt": 2.33, "measured_on": "OLMo-7B-hf, canon only"},
}
FAMILY = {
    "mistralai/Mistral-Nemo-Instruct-2407": "nemo-12b",
    "Vikhrmodels/Vikhr-Nemo-12B-Instruct-R-21-09-24": "nemo-12b",
    "yandex/YandexGPT-5-Lite-8B-pretrain": "yandex-8b",
    "yandex/YandexGPT-5-Lite-8B-instruct": "yandex-8b",
    "Qwen/Qwen2.5-7B-Instruct": "qwen-7b",
    # the two adaptations share Qwen2.5-7B's architecture; their own first
    # session replaces this assumption with a measurement
    "t-tech/T-lite-it-1.0": "qwen-7b",
    "RefalMachine/ruadapt_qwen2.5_7B_ext_u48_instruct": "qwen-7b",
    "allenai/OLMo-7B-hf": "olmo-7b",
}
# a model never measured is priced as the slowest family measured, and the
# output says so; its first session corrects the figure
UNMEASURED = "nemo-12b"
BUDGET = {"header": 1000, "feature": 64}
SAFETY = 1.15
PREFIX_ROWS = 8
FEATURE_ROWS = 6          # five few-shot observations and the query, as "column = value"
HEADER_WINDOW_CHARS = 500


def per_query(cost, n_in, n_out):
    return (cost["fixed"] + cost["per_token"] * n_out
            + cost["per_token_per_1k_context"] * n_out * (n_in + n_out / 2) / 1000
            + cost["per_1k_prompt"] * n_in / 1000)


def count(tokenizer, text):
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def shapes(dataset, test, tokenizer, cache={}):
    """(prompt tokens, answer budget) of one query of this cell, from the data."""
    key = (dataset, test, id(tokenizer))
    if key in cache:
        return cache[key]
    record = load_registry()[dataset[:-4]]
    path = os.path.join(ROOT, record["variants"]["raw"]["path"])
    text = open(path, encoding="utf-8").read()
    lines = [line for line in text.split("\n") if line]
    rows = lines[1:401]
    row_tokens = statistics.mean(count(tokenizer, r) for r in rows)
    row_chars = statistics.mean(len(r) for r in lines[1:2001])
    if test == "header":
        shape = (count(tokenizer, text[:HEADER_WINDOW_CHARS]), BUDGET["header"])
    elif test == "feature":
        import csv
        names = next(csv.reader([lines[0]]))
        rendered = [", ".join(f"{n} = {v}" for n, v in zip(names, next(csv.reader([r]))))
                    for r in rows[:100]]
        shape = (FEATURE_ROWS * statistics.mean(count(tokenizer, r) for r in rendered),
                 BUDGET["feature"])
    else:  # row completion, and first token, which the runner asks the same way
        shape = ((PREFIX_ROWS + 1) * row_tokens, 1 + row_chars)
    cache[key] = shape
    return shape


def price_cells(cells, model, tokenizer):
    cost = COST[FAMILY.get(model, UNMEASURED)]
    lines, total = [], 0.0
    for dataset, test, queries in cells:
        n_in, n_out = shapes(dataset, test, tokenizer)
        seconds = SAFETY * queries * per_query(cost, n_in, n_out)
        total += seconds
        lines.append(f"  {dataset:28s} {test:12s} {queries:4d} queries  prompt {n_in:5.0f}  "
                     f"budget {n_out:5.0f}  {seconds / 60:6.0f} min")
    return lines, total


def load_tokenizer(model):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model)


def validate():
    """Every completed completion-mode run of blocks B and C against its bound."""
    runs = []
    for path in sorted(glob.glob(os.path.join(ROOT, "results", "gateA_*_completion_2026*.json"))):
        if any(stamp in path for stamp in ("20260924T", "20260925T")):  # void Y1 attempts
            continue
        run = json.load(open(path, encoding="utf-8"))
        cells = [(r["dataset_key"], r["test"], r.get("n") or r.get("requested_queries") or 4)
                 for r in run["results"] if "error" not in r]
        actual = sum(r.get("seconds", 0) for r in run["results"] if "error" not in r)
        runs.append((os.path.basename(path), run["model"], run["plan"], cells, actual))
    tokenizers = {}
    worst = 0.0
    print(f"planning figure = budget bound x {SAFETY}")
    print(f"{'run':74s} {'actual h':>8s} {'plan h':>8s} {'actual/plan':>12s}")
    for name, model, plan, cells, actual in runs:
        if model not in tokenizers:
            tokenizers[model] = load_tokenizer(model)
        _, bound = price_cells(cells, model, tokenizers[model])
        ratio = actual / bound if bound else 0.0
        worst = max(worst, ratio)
        print(f"{name[:74]:74s} {actual / 3600:8.2f} {bound / 3600:8.2f} {ratio:12.2f}"
              + ("   EXCEEDS THE BOUND" if ratio > 1 else ""))
    print(f"\nlargest actual/plan: {worst:.2f}")
    return 0 if worst <= 1 else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("plans", nargs="*")
    ap.add_argument("--model", default="mistralai/Mistral-Nemo-Instruct-2407",
                    help="the model to price for: its tokenizer counts the tokens, its "
                         "measured family sets the cost (unmeasured: priced as the slowest)")
    ap.add_argument("--validate", action="store_true")
    args = ap.parse_args()
    if args.validate:
        return validate()

    tokenizer = load_tokenizer(args.model)
    family = FAMILY.get(args.model)
    if family is None:
        print(f"{args.model} has no measured cost; priced as {UNMEASURED} "
              f"({COST[UNMEASURED]['measured_on']})")
    for plan in args.plans:
        lines, total = price_cells(PLANS[plan], args.model, tokenizer)
        print(f"\n{plan}: {sum(q for _, _, q in PLANS[plan])} queries; planning figure = "
              f"every answer runs to its budget, x {SAFETY}")
        print("\n".join(lines))
        print(f"  {'total':28s} {total / 3600:6.1f} h, plus 5-15 min to download and load "
              "the model")
    return 0


if __name__ == "__main__":
    sys.exit(main())
