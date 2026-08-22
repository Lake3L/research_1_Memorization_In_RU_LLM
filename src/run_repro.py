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
from detectability import minimum_detectable_rate  # noqa: E402
from metrics import header_verdict, infer_separator, normalise_numbers  # noqa: E402
from mock_llm import PerfectMemorizer, format_echo_mock  # noqa: E402

# max_tokens per test. The tight caps below were chosen to make a $5 OpenAI
# budget stretch; on open weights that reason does not apply, and on the header
# test the cap is not merely conservative — the verdict counts how many rows the
# model reproduced, so capping the answer caps the statistic. AMENDMENT_4 §2
# restores the library default for open-weight runs.
MAX_TOKENS = {"header": 300, "row": 100, "feature": 60, "first_token": 100}
MAX_TOKENS_REFERENCE = {"header": 1000, "row": 1000, "feature": 1000,
                        "first_token": 1000}

# The parameters Bordt et al. used for the open models of Table 3, read from
# their own code (colm-2024-paper-code/notebooks/memorization-tests.ipynb) rather
# than from the library defaults, which differ. See AMENDMENT_4 §1.
PROTOCOL = {
    "reference": {"num_prefix_rows": 8, "few_shot_row": 5, "completion_length": 350,
                  "max_tokens": MAX_TOKENS_REFERENCE},
    "legacy": {"num_prefix_rows": 10, "few_shot_row": 7, "completion_length": 500,
               "max_tokens": MAX_TOKENS},
}

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Reference values from the paper. Row/feature completion: Table 5 (p. 21).
# First token: Table 6 (p. 21). Header: Table 2 (p. 4), pass/fail.
PAPER = {
    ("iris.csv", "row"): {"gpt-3.5": (35, 136), "gpt-4": (125, 136)},
    ("uci-wine.csv", "row"): {"gpt-3.5": (16, 164), "gpt-4": (84, 164)},
    ("openml-diabetes.csv", "row"): {"gpt-3.5": (18, 250), "gpt-4": (79, 250)},
    ("titanic-train.csv", "row"): {"gpt-3.5": (194, 250), "gpt-4": (222, 250)},
    ("adult-train.csv", "row"): {"gpt-3.5": (0, 250), "gpt-4": (0, 250)},
    ("california-housing.csv", "row"): {"gpt-3.5": (0, 250), "gpt-4": (0, 250)},
    ("uci-wine.csv", "feature"): {"gpt-3.5": (77, 178), "gpt-4": (131, 178)},
    ("openml-diabetes.csv", "feature"): {"gpt-3.5": (237, 250), "gpt-4": (243, 250)},
    ("titanic-train.csv", "feature"): {"gpt-3.5": (238, 250), "gpt-4": (236, 250)},
    ("adult-train.csv", "feature"): {"gpt-3.5": (0, 250), "gpt-4": (0, 250)},
    ("california-housing.csv", "feature"): {"gpt-3.5": (0, 250), "gpt-4": (1, 250)},
    ("iris.csv", "first_token"): {"gpt-3.5": (88, 136), "gpt-4": (131, 136), "baseline": 0.50},
    ("openml-diabetes.csv", "first_token"): {"gpt-3.5": (42, 250), "gpt-4": (95, 250), "baseline": 0.25},
    ("adult-train.csv", "first_token"): {"gpt-3.5": (59, 250), "gpt-4": (68, 250), "baseline": 0.26},
    ("iris.csv", "header"): {"gpt-3.5": "pass", "gpt-4": "pass"},
    ("uci-wine.csv", "header"): {"gpt-3.5": "pass", "gpt-4": "pass"},
    ("openml-diabetes.csv", "header"): {"gpt-3.5": "pass", "gpt-4": "pass"},
    ("titanic-train.csv", "header"): {"gpt-3.5": "pass", "gpt-4": "pass"},
    ("adult-train.csv", "header"): {"gpt-3.5": "pass", "gpt-4": "pass"},
    ("california-housing.csv", "header"): {"gpt-3.5": "pass", "gpt-4": "pass"},
}

# The paper's feature choices (Table 5 caption). Iris is deliberately absent:
# it has no high-entropy feature, which is why the paper runs no feature
# completion test on it.
FEATURES = {
    "uci-wine.csv": "malic_acid",
    "openml-diabetes.csv": "DiabetesPedigreeFunction",
    "titanic-train.csv": "Name",
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
    # PREREGISTRATION.md §8, second half: the adapted HF pipeline has to
    # reproduce the English result of the unmodified one. Every cell of the
    # OpenAI gate above is repeated so the two runs are directly comparable,
    # with three additions the paid gate could not afford: Kaggle Titanic (the
    # paper's strongest row-completion signal, 194/250) and row completion on
    # wine and california-housing, which round out the contrast between the
    # memorized small classics and the large datasets nobody memorizes.
    # Open 7-8B models are expected to be weaker than GPT-4 at extraction, so
    # the first-token test — the most sensitive of the four — runs everywhere
    # the paper reports a baseline for it.
    "gate_hf": [
        ("iris.csv", "header", 4), ("uci-wine.csv", "header", 4),
        ("openml-diabetes.csv", "header", 4), ("titanic-train.csv", "header", 4),
        ("adult-train.csv", "header", 4), ("california-housing.csv", "header", 4),
        ("iris.csv", "row", 50), ("titanic-train.csv", "row", 25),
        ("uci-wine.csv", "row", 25), ("openml-diabetes.csv", "row", 25),
        ("adult-train.csv", "row", 25), ("california-housing.csv", "row", 25),
        ("titanic-train.csv", "feature", 50), ("openml-diabetes.csv", "feature", 50),
        ("uci-wine.csv", "feature", 50), ("adult-train.csv", "feature", 50),
        ("california-housing.csv", "feature", 50),
        ("iris.csv", "first_token", 50), ("openml-diabetes.csv", "first_token", 50),
        ("adult-train.csv", "first_token", 50),
    ],
    # AMENDMENT_4 §5: dataset-bounded query counts, set by the power analysis in
    # src/power_h1b.py rather than by what fits. Row completion cannot ask more
    # questions than the file has rows, so iris and wine are exhausted (142 and
    # 170 with eight prefix rows) and the rest are capped at the paper's 250.
    "h1b": [
        ("iris.csv", "header", 4), ("uci-wine.csv", "header", 4),
        ("openml-diabetes.csv", "header", 4), ("titanic-train.csv", "header", 4),
        ("adult-train.csv", "header", 4), ("california-housing.csv", "header", 4),
        ("iris.csv", "row", 142), ("uci-wine.csv", "row", 170),
        ("openml-diabetes.csv", "row", 250), ("titanic-train.csv", "row", 250),
        ("adult-train.csv", "row", 250), ("california-housing.csv", "row", 250),
        ("iris.csv", "first_token", 142), ("openml-diabetes.csv", "first_token", 250),
        ("adult-train.csv", "first_token", 250),
        ("uci-wine.csv", "feature", 170), ("openml-diabetes.csv", "feature", 250),
        ("titanic-train.csv", "feature", 250), ("adult-train.csv", "feature", 250),
        ("california-housing.csv", "feature", 250),
    ],
    # The prompting-mode probe of AMENDMENT_4 §3, cheap enough to run in both
    # modes back to back: row completion only, on the four datasets where the
    # paper reports a non-zero count for anyone, plus one negative control.
    "probe": [
        ("iris.csv", "row", 142), ("uci-wine.csv", "row", 170),
        ("openml-diabetes.csv", "row", 250), ("titanic-train.csv", "row", 250),
        ("adult-train.csv", "row", 100),
    ],
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


def digits_per_row(csv_file):
    """Ward et al.'s covariate: long digit strings are harder to reproduce.

    Two datasets with the same rate are not the same feat if one carries eight
    digits per row and the other ninety-six, so AMENDMENT_5 §1 requires this
    beside every count rather than in a separate table nobody reads.
    """
    rows = utils.load_csv_rows(csv_file)
    if not rows:
        return None
    return round(sum(sum(ch.isdigit() for ch in row) for row in rows) / len(rows), 1)


def duplicate_rate(csv_file):
    rows = utils.load_csv_rows(csv_file)
    return 1 - len(set(rows)) / len(rows)


def response_diagnostics(suffixes, responses):
    """Is the model even trying to produce CSV rows?

    A count of zero exact matches has two very different causes: the model never
    saw the data, or our adapter never got a well-formed answer out of it (wrong
    chat template, a refusal, a chatty preamble, truncation). Nothing in the
    count distinguishes them, so we measure the shape of the answers separately
    from their content.

    `well_formed_rate` compares field counts, not content. `mean_normalized_
    levenshtein` and `near_match_rate` are the approximate-memorization metric of
    Ward et al. (PREREGISTRATION.md §2, and AMENDMENT_3_H1B_OUTCOMES.md, where it
    becomes a secondary outcome of H1b): a model that reproduces a row up to one
    digit has not "failed to memorize" in any useful sense, and an exact-match
    count alone would record that as a zero.

    Number formats are canonicalised before the distance is taken, so that
    `17.500` against `17.5` is a distance of zero. The exact-match count above is
    deliberately *not* normalised: it is the verbatim measure.
    """
    import jellyfish

    def first_line(text):
        for line in str(text).strip().split("\n"):
            if line.strip():
                return line.strip()
        return ""

    well_formed, distances = 0, []
    for suffix, response in zip(suffixes, responses):
        truth, got = str(suffix).strip(), first_line(response)
        sep = infer_separator(truth)
        if truth.count(sep) > 0 and got.count(sep) == truth.count(sep):
            well_formed += 1
        if truth or got:
            a, b = normalise_numbers(truth, sep), normalise_numbers(got, sep)
            distances.append(jellyfish.levenshtein_distance(a, b)
                             / max(len(a), len(b), 1))
    n = len(distances)
    return {
        "well_formed_rate": round(well_formed / len(responses), 4) if responses else 0.0,
        "mean_normalized_levenshtein": round(sum(distances) / n, 4) if n else None,
        "near_match_rate": round(sum(1 for d in distances if d <= 0.1) / n, 4) if n else None,
    }


def dataset_key(csv_file):
    """The dataset a file belongs to, whatever serialisation it is.

    `data/canon/variants/uci-wine__cp1251_semicolon.csv` and a bare `uci-wine.csv`
    are the same dataset, and the paper's reference numbers and designated
    features are keyed by dataset, not by path. Getting this wrong is silent: a
    lookup miss makes tabmemcheck pick its own feature, and the result is then
    not comparable with the published table it is printed next to.
    """
    stem = os.path.splitext(os.path.basename(csv_file))[0]
    return stem.split("__")[0] + ".csv"


def designated_feature(csv_file):
    """The feature the paper tested, or the most unique one if it named none."""
    feature = FEATURES.get(dataset_key(csv_file))
    if feature is not None:
        return feature, "paper"
    from tabmemcheck import analysis
    feature, _ = analysis.find_most_unique_feature(csv_file)
    return feature, "most_unique"


def run_one(llm, csv_file, test, num_queries, seed, protocol="reference"):
    """Run one test, returning a result dict. Output of tabmemcheck is captured.

    Counts only. The preregistered decision rules (§5) — binomial tests against
    the best of mode/LR/GBT baselines, Holm correction within a hypothesis
    family — are applied offline, over these counts and the JSONL call logs, so
    that a scoring rule can be revised without re-running rented compute.
    """
    rng = np.random.default_rng(seed)
    buf = io.StringIO()
    settings = PROTOCOL[protocol]
    result = {"dataset": csv_file, "dataset_key": dataset_key(csv_file),
              "test": test, "requested_queries": num_queries, "protocol": protocol,
              "chat_mode": bool(getattr(llm, "chat_mode", True))}
    tabmem.config.max_tokens = settings["max_tokens"].get(test, 1000)

    with redirect_stdout(buf):
        if test == "header":
            try:
                prompt, completion, response = tabmem.header_test(
                    csv_file, llm, rng=rng,
                    completion_length=settings["completion_length"])
            except (UnboundLocalError, NameError):
                # tabmemcheck's header_test tracks the best of four splits with a
                # sentinel it only overwrites when a response is non-empty, so a
                # model that returns nothing on all four splits leaves the result
                # variable unbound (functions.py, header_test). That is a failing
                # header test, not a crash — but the distinction between "wrote
                # nothing" and "wrote the wrong thing" matters, so it is recorded
                # rather than folded into an ordinary fail. The four prompts and
                # their empty responses are in the JSONL call log.
                result.update(verdict="fail", rows_recovered=0, prefix_match_chars=0,
                              note="model returned no text on any of the four splits")
            else:
                # Bordt's criterion is "the model completes at least the next
                # row", judged by eye from a Levenshtein-coloured printout.
                # metrics.py reports both its strict and its automatable form;
                # the verdict is the latter (RESULTS_GATE.md §0).
                result.update(header_verdict(completion, response))
                result["response_head"] = response[:200]
                result["true_head"] = completion[:200]

        elif test == "row":
            suffixes, responses = tabmem.row_completion_test(
                csv_file, llm, num_queries=num_queries, rng=rng,
                num_prefix_rows=settings["num_prefix_rows"],
                few_shot=settings["few_shot_row"], print_levenshtein=False
            )
            n = len(responses)
            k = sum(1 for s, r in zip(suffixes, responses) if s.strip() in r.strip())
            base = duplicate_rate(csv_file)
            result.update(matches=k, n=n, rate=k / n if n else 0.0,
                          baseline_rate=base,
                          p_value=float(stats.binomtest(k, n, max(base, 1e-9),
                                                        alternative="greater").pvalue) if n else None,
                          # AMENDMENT_5 §1: a zero is not a finding until the
                          # smallest effect it could have excluded is stated
                          minimum_detectable_rate=minimum_detectable_rate(base, n),
                          digits_per_row=digits_per_row(csv_file),
                          **response_diagnostics(suffixes, responses))

        elif test == "feature":
            feature, chosen_by = designated_feature(csv_file)
            try:
                values, responses = tabmem.feature_completion_test(
                    csv_file, llm, feature_name=feature, num_queries=num_queries, rng=rng
                )
            except KeyError:
                # tabmemcheck parses the answers as "Feature = Value" and indexes
                # the resulting frame by feature name; when not one answer parses,
                # the column is absent. That is zero matches with an important
                # caveat attached — the zero may be a formatting failure rather
                # than absence of memorization — so it is recorded as a count with
                # the caveat, not discarded as an error.
                result.update(feature=feature, feature_chosen_by=chosen_by,
                              matches=0, n=num_queries, rate=0.0,
                              note="no response parsed as 'Feature = Value'; "
                                   "a zero here may be a format failure, see the call log")
                result["stdout"] = ANSI.sub("", buf.getvalue())[-2000:]
                return result
            n = len(responses)
            k = sum(1 for v, r in zip(values, responses)
                    if str(v).strip() == str(r).strip())
            df = utils.load_csv_df(csv_file)
            mode_rate = float(df[feature].astype(str).value_counts(normalize=True).iloc[0])
            result.update(feature=feature, feature_chosen_by=chosen_by,
                          matches=k, n=n, rate=k / n if n else 0.0,
                          mode_baseline=mode_rate,
                          p_value=float(stats.binomtest(k, n, max(mode_rate, 1e-9),
                                                        alternative="greater").pvalue) if n else None,
                          minimum_detectable_rate=minimum_detectable_rate(mode_rate, n),
                          digits_per_row=digits_per_row(csv_file))

        elif test == "first_token":
            tabmem.first_token_test(csv_file, llm, num_queries=num_queries, rng=rng,
                                    num_prefix_rows=settings["num_prefix_rows"],
                                    few_shot=settings["few_shot_row"])
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
        ref = PAPER.get((dataset_key(csv_file), test), {})
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
