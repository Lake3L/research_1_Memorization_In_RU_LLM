"""The same answers, scored two ways: exact match, and how close the misses were.

AMENDMENT_3 §2 added the graded measure as a secondary outcome for H1b, on the
argument that five identical zeros in the primary outcome can be five different
numbers underneath. That argument is sound and this script is how it is checked
rather than assumed — because the graded measure is not automatically better, and
on the 12B pilot it points the other way from the binary one.

It prints, for one dataset and a pair of models:

  the binary comparison  — exact matches and a Fisher exact test, which is the
      preregistered primary outcome and stays byte-exact;
  the graded comparison  — normalised edit distance after number canonicalisation,
      with Mann-Whitney, which is the secondary outcome;
  the distribution        — how many answers were exact, near, partial or
      unrelated, because a mean over a bimodal quantity summarises nothing.

The distribution is the part that matters. A model can be higher on exact matches
and worse on average distance at the same time, by being right more often and
wrong more badly, and neither number alone would show it.

Usage:
  python src/compare_measures.py --dataset iris.csv \\
      --base results/calls_..._Mistral-Nemo....jsonl \\
      --adapted results/calls_..._Vikhr-Nemo....jsonl
"""

import argparse
import json
import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset_registry import load_registry  # noqa: E402
from metrics import infer_separator, normalise_numbers  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BANDS = [(0.0, 1e-9, "exact"), (1e-9, 0.1, "near (<=0.1)"),
         (0.1, 0.5, "partial"), (0.5, 1.01, "unrelated")]


def first_line(text):
    for line in str(text).strip().split("\n"):
        if line.strip():
            return line.strip()
    return ""


def score(log_path, dataset, csv_path):
    """Exact-match flags and normalised distances, rebuilt from the raw log."""
    import jellyfish
    from tabmemcheck import utils

    rows = utils.load_csv_rows(csv_path)
    index = {row: i for i, row in enumerate(rows)}
    exact, distances = [], []
    for call in (json.loads(l) for l in open(log_path, encoding="utf-8")):
        if call.get("test") != "row" or call.get("dataset") != dataset:
            continue
        prompt = [m for m in call["messages"] if m["role"] == "user"][-1]["content"]
        i = index.get(prompt.strip().split("\n")[-1].strip())
        if i is None or i + 1 >= len(rows):
            continue
        truth, got = rows[i + 1], first_line(call["response"])
        # the primary outcome is byte-exact and stays that way
        exact.append(truth.strip() in str(call["response"]).strip())
        sep = infer_separator(truth)
        a, b = normalise_numbers(truth, sep), normalise_numbers(got, sep)
        distances.append(jellyfish.levenshtein_distance(a, b) / max(len(a), len(b), 1))
    return np.array(exact), np.array(distances)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapted", required=True)
    ap.add_argument("--dataset", default="iris.csv")
    ap.add_argument("--variant", default="raw")
    args = ap.parse_args()

    registry = load_registry()
    name = args.dataset[:-4] if args.dataset.endswith(".csv") else args.dataset
    csv_path = os.path.join(ROOT, registry[name]["variants"][args.variant]["path"])

    base_exact, base_dist = score(args.base, args.dataset, csv_path)
    adapted_exact, adapted_dist = score(args.adapted, args.dataset, csv_path)

    hb, ha = int(base_exact.sum()), int(adapted_exact.sum())
    nb, na = len(base_exact), len(adapted_exact)
    print(f"{args.dataset}: base n={nb}, adapted n={na}\n")

    print("PRIMARY — exact match, byte-exact (PREREGISTRATION.md §5)")
    print(f"  base {hb}/{nb} = {hb/nb:.3f}   adapted {ha}/{na} = {ha/na:.3f}")
    odds, p = stats.fisher_exact([[hb, nb - hb], [ha, na - ha]])
    print(f"  Fisher exact p = {p:.3f}   "
          f"direction: {'adapted higher' if ha/na > hb/nb else 'base higher' if ha/na < hb/nb else 'equal'}\n")

    print("SECONDARY — normalised edit distance, lower is closer (AMENDMENT_3 §2)")
    print(f"  base    mean {base_dist.mean():.3f}  median {np.median(base_dist):.3f}")
    print(f"  adapted mean {adapted_dist.mean():.3f}  median {np.median(adapted_dist):.3f}")
    u_p = stats.mannwhitneyu(base_dist, adapted_dist, alternative="two-sided").pvalue
    print(f"  Mann-Whitney p = {u_p:.3f}   "
          f"direction: {'base closer' if base_dist.mean() < adapted_dist.mean() else 'adapted closer'}\n")

    print("DISTRIBUTION — the part a mean hides")
    print(f"  {'band':16s} {'base':>6s} {'adapted':>8s}")
    for low, high, label in BANDS:
        b = int(((base_dist >= low) & (base_dist < high)).sum())
        a = int(((adapted_dist >= low) & (adapted_dist < high)).sum())
        print(f"  {label:16s} {b:6d} {a:8d}")

    if (ha / na > hb / nb) != (adapted_dist.mean() < base_dist.mean()):
        print("\nNote: the two measures disagree in direction. That is not a reason to")
        print("prefer either — it means one model is right more often and wrong more")
        print("badly, and the distribution above is the finding, not either summary.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
