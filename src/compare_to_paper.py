"""Compare our reproduction runs against the published tables and decide the
positive-control gate of PREREGISTRATION.md §8.

Emits a markdown table: our count, our rate with a 95% Clopper-Pearson
interval, the paper's rate, and whether the paper's rate falls inside our
interval. Sample sizes are smaller than the paper's, so wide intervals are
expected and are reported rather than hidden.
"""

import glob
import json
import os
import sys

from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PAPER_RATE = {  # (dataset, test) -> {model_family: (k, n)}
    ("iris.csv", "row"): {"gpt-3.5": (35, 136), "gpt-4": (125, 136)},
    ("uci-wine.csv", "row"): {"gpt-3.5": (16, 164), "gpt-4": (84, 164)},
    ("openml-diabetes.csv", "row"): {"gpt-3.5": (18, 250), "gpt-4": (79, 250)},
    ("adult-train.csv", "row"): {"gpt-3.5": (0, 250), "gpt-4": (0, 250)},
    ("california-housing.csv", "row"): {"gpt-3.5": (0, 250), "gpt-4": (0, 250)},
    ("uci-wine.csv", "feature"): {"gpt-3.5": (77, 178), "gpt-4": (131, 178)},
    ("openml-diabetes.csv", "feature"): {"gpt-3.5": (237, 250), "gpt-4": (243, 250)},
    ("adult-train.csv", "feature"): {"gpt-3.5": (0, 250), "gpt-4": (0, 250)},
    ("california-housing.csv", "feature"): {"gpt-3.5": (0, 250), "gpt-4": (1, 250)},
    ("iris.csv", "first_token"): {"gpt-3.5": (88, 136), "gpt-4": (131, 136)},
    ("openml-diabetes.csv", "first_token"): {"gpt-3.5": (42, 250), "gpt-4": (95, 250)},
    ("adult-train.csv", "first_token"): {"gpt-3.5": (59, 250), "gpt-4": (68, 250)},
}


def family(served_model: str):
    """Which published column, if any, this model may be compared against.

    Only the checkpoints the paper actually ran have a reference column.
    gpt-4o and gpt-4o-mini are different models despite the shared prefix;
    comparing them to the GPT-4 column would be meaningless, so they get None
    and are reported as an extension rather than a reproduction.
    """
    if served_model.startswith(("gpt-4-0613", "gpt-4-0125", "gpt-4-32k")):
        return "gpt-4"
    if served_model.startswith("gpt-3.5"):
        return "gpt-3.5"
    return None


def ci(k, n):
    if n == 0:
        return (0.0, 0.0)
    lo, hi = stats.binomtest(k, n).proportion_ci(confidence_level=0.95)
    return (lo, hi)


def rescore_for(run_path):
    """The header re-scoring belonging to THIS run, never another run's."""
    p = os.path.join(os.path.dirname(run_path),
                     "header_rescored_" + os.path.basename(run_path)[len("repro_"):])
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def report(paths):
    lines = []
    verdicts = []
    for path in paths:
        d = json.load(open(path, encoding="utf-8"))
        header_rescore = rescore_for(path)
        served = list(d["llm_summary"].get("served_models", {}))
        if not served:
            # runs made before served-model tracking: read it from the chatlog
            log = os.path.join(os.path.dirname(path),
                               "chatlog_" + os.path.basename(path)[len("repro_"):]
                               .replace(".json", ".jsonl"))
            if os.path.exists(log):
                with open(log, encoding="utf-8") as f:
                    served = [json.loads(f.readline())["model"]]
        served_model = served[0] if served else d["model_requested"]
        fam = family(served_model)
        role = ("reproduction" if fam else
                "extension — no published column for this model, not counted below")
        lines.append(f"\n### Requested `{d['model_requested']}` → served `{served_model}` "
                     f"(seed {d['seed']}, {d['llm_summary']['calls']} calls) — {role}\n")
        lines.append("| test | dataset | ours | rate [95% CI] | paper | paper rate | paper in CI |")
        lines.append("|---|---|---|---|---|---|---|")
        for r in d["results"]:
            if "error" in r:
                lines.append(f"| {r['test']} | {r['dataset']} | ERROR | — | — | — | — |")
                continue
            key = (r["dataset"], r["test"])
            if r["test"] == "header":
                ours = r.get("verdict")
                if header_rescore and r["dataset"] in header_rescore:
                    ours = header_rescore[r["dataset"]]["verdict"]
                    ours += f" ({header_rescore[r['dataset']]['max_rows_recovered']} rows)"
                if fam is None:
                    lines.append(f"| header | {r['dataset']} | {ours} | — | — | — | n/a |")
                    continue
                agree = "yes" if str(ours).startswith("pass") else "NO"
                lines.append(f"| header | {r['dataset']} | {ours} | — | pass | — | {agree} |")
                verdicts.append((r["test"], r["dataset"], agree == "yes"))
                continue
            k, n = r.get("matches"), r.get("n")
            if k is None or not n:
                lines.append(f"| {r['test']} | {r['dataset']} | (unparsed) | — | — | — | — |")
                continue
            lo, hi = ci(k, n)
            pk, pn = PAPER_RATE.get(key, {}).get(fam, (None, None)) if fam else (None, None)
            if pk is None:
                lines.append(f"| {r['test']} | {r['dataset']} | {k}/{n} | "
                             f"{k/n:.2f} [{lo:.2f}, {hi:.2f}] | — | — | — |")
                continue
            prate = pk / pn
            inside = lo <= prate <= hi
            verdicts.append((r["test"], r["dataset"], inside))
            lines.append(f"| {r['test']} | {r['dataset']} | {k}/{n} | "
                         f"{k/n:.2f} [{lo:.2f}, {hi:.2f}] | {pk}/{pn} | {prate:.2f} | "
                         f"{'yes' if inside else 'NO'} |")
    agree = sum(1 for _, _, ok in verdicts if ok)
    lines.append(f"\n**Cells consistent with the paper: {agree}/{len(verdicts)}**")
    return "\n".join(lines)


if __name__ == "__main__":
    paths = sys.argv[1:] or sorted(glob.glob(os.path.join(ROOT, "results", "repro_*live*.json")))
    print(report(paths))
