"""How many queries per cell before a base-vs-adapted difference decides anything.

PREREGISTRATION.md §6 makes H1b a paired comparison between a base model and its
Russian adaptation. The 12B pilot put Mistral-Nemo at 4/50 and Vikhr-Nemo at 7/50
on iris row completion — a difference in the direction the hypothesis cares about
and, at n=50, a difference that a coin could produce. This computes what it would
take to call it, so that the next run is sized before it is paid for rather than
interpreted after.

Two constraints fight each other and both are reported:

  statistics — the smaller the true difference, the more queries it takes;
  the data   — row completion cannot ask more questions than the file has rows.
               Iris has 150 rows, so with 8 rows of prefix there are 142 possible
               queries and no protocol can buy more. Bordt et al. hit the same
               wall and report 136 for iris where they report 250 elsewhere.

Usage:
  python src/power_h1b.py
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scipy import stats  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Rows in each canon file, and therefore the ceiling on row-completion queries
# once the prefix is taken out. Read from the registry rather than typed.
PREFIX_ROWS = 8   # the authors' setting for open models


def dataset_ceilings(prefix_rows=PREFIX_ROWS):
    import json
    registry = json.load(open(os.path.join(ROOT, "data", "registry.json"),
                              encoding="utf-8"))
    return {name: rec["n_rows"] - prefix_rows
            for name, rec in registry.items() if rec["group"] == "canon"}


def cohens_h(p1, p2):
    return abs(2 * math.asin(math.sqrt(p2)) - 2 * math.asin(math.sqrt(p1)))


def n_for_power(p1, p2, power=0.8, alpha=0.05):
    """Queries per arm for a two-proportion comparison at the given power."""
    h = cohens_h(p1, p2)
    if h == 0:
        return math.inf
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)
    return math.ceil(((z_alpha + z_beta) / h) ** 2)


def achieved_power(p1, p2, n, alpha=0.05):
    h = cohens_h(p1, p2)
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    return float(stats.norm.cdf(h * math.sqrt(n) - z_alpha))


def main():
    print("Observed in the 12B pilot (one variant, one seed, chat prompting):")
    print("  iris row completion — Mistral-Nemo 4/50 = 0.08, Vikhr-Nemo 7/50 = 0.14\n")

    print("Queries per arm needed to call a base-vs-adapted difference, by how big")
    print("the true difference is (alpha 0.05, power 0.80, two-sided):\n")
    print(f"  {'base':>6s} {'adapted':>8s} {'Cohen h':>8s} {'n per arm':>10s}")
    for p1, p2 in [(0.08, 0.14), (0.08, 0.16), (0.08, 0.20), (0.08, 0.24),
                   (0.26, 0.40), (0.26, 0.50), (0.10, 0.30)]:
        print(f"  {p1:6.2f} {p2:8.2f} {cohens_h(p1, p2):8.3f} "
              f"{n_for_power(p1, p2):10d}")

    ceilings = dataset_ceilings()
    print(f"\nCeiling on row-completion queries per dataset "
          f"(rows minus {PREFIX_ROWS} prefix rows):\n")
    for name, ceiling in sorted(ceilings.items(), key=lambda kv: kv[1]):
        note = ""
        if ceiling < n_for_power(0.08, 0.14):
            note = "  <- cannot reach 80% power for the observed effect"
        print(f"  {name:24s} {ceiling:6d}{note}")

    iris_ceiling = ceilings.get("iris", 142)
    print(f"\nOn iris, the only dataset where anything extracts, the ceiling is "
          f"{iris_ceiling}.")
    print(f"At n={iris_ceiling} the achieved power for 0.08 vs 0.14 is "
          f"{achieved_power(0.08, 0.14, iris_ceiling):.0%}, and for 0.08 vs 0.20 it is "
          f"{achieved_power(0.08, 0.20, iris_ceiling):.0%}.")
    print("\nSo: exhausting iris is necessary and, for a difference as small as the")
    print("pilot's, still not sufficient. A decisive H1b needs either a larger true")
    print("difference — which a more sensitive probe would produce if the signal is")
    print("there — or more datasets off the floor. Both are what the completion-mode")
    print("probe in AMENDMENT_4 is for; neither is bought by more seeds.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
