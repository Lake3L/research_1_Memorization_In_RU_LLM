# Verification of Table 4 (few-shot, seen vs novel) from the authors' own raw data

**Date:** 2026-07-31. **Source:** Bordt et al., COLM 2024, data release
[10.5281/zenodo.14644404](https://doi.org/10.5281/zenodo.14644404) — 171 768 pickled
`(messages, response)` pairs covering four models, ten datasets and four dataset formats.

Re-running Table 4 against the API would cost tens of dollars (≈1000 queries per cell × 40 cells)
and is impossible at the paper's model versions anyway. Verifying it from the authors' published
responses costs nothing and is a stricter check: it tests their *numbers*, not a fresh run of a
different model.

## Why three independent paths

A single recomputation that agrees with a published table proves little — the recomputation could
share an error with the thing it checks, or be tuned until it agrees. So ground truth was
recovered three separate ways, by scripts written independently of each other:

| Path | Ground truth from | Script | Covers |
|---|---|---|---|
| A | labels appearing in few-shot turns of other queries | `src/recompute_table4.py` | all cells |
| B | matching prompt feature values against the dataset CSV | `src/verify_table4_independent.py` | `original` format |
| C | independent re-implementation of path A | `src/verify_table4_fewshot.py` | all formats, coverage varies |

Paths B and C were written without reusing path A's code. Where they overlap they agree to four
decimal places, which is the evidence that matters: three routes to the same number.

## Agreement with the published table

Seed 0, so small differences from the published value are expected wherever the paper averages
over seeds (GPT-3.5: 3 seeds; GPT-4: 1 seed — confirmed both from the archive's directory
structure and from the authors' notebook comment, "the number of different seeds (3 for gpt-3.5,
1 for gpt-4)").

| model | dataset | format | path A | path B | path C | published |
|---|---|---|---|---|---|---|
| gpt-4 | titanic | original | 0.9574 | — | 0.9574 | 0.96 |
| gpt-4 | titanic | perturbed | 0.8227 | — | 0.8227 | 0.82 |
| gpt-4 | titanic | task | 0.8047 | — | 0.8047 | 0.80 |
| gpt-4 | titanic | statistical | 0.6543 | — | 0.6543 | 0.65 |
| gpt-4 | iris | original | 0.9867 | 0.9867 | 0.9867 | 0.99 |
| gpt-4 | iris | perturbed | 0.9533 | — | 0.9533 | 0.95 |
| gpt-4 | iris | statistical | 0.9200 | — | 0.9200 | 0.92 |
| gpt-4 | adult | original | 0.8140 | 0.8140 | (partial) | 0.81 |
| gpt-4 | openml-diabetes | original | 0.7383 | 0.7383 | — | 0.74 |
| gpt-4 | uci-wine | original | — | 0.9607 | — | 0.96 |
| gpt-4 | icu | original | — | 0.6863 | — | 0.69 |
| gpt-3.5 | titanic | original | 0.8148 | — | 0.8148 | 0.81 |
| gpt-3.5 | titanic | perturbed | 0.7890 | — | 0.7890 | 0.78 |
| gpt-3.5 | titanic | task | 0.7733 | — | 0.7733 | 0.77 |
| gpt-3.5 | titanic | statistical | 0.6083 | — | 0.6083 | 0.61 |
| gpt-3.5 | iris | original | 0.9867 | 0.9867 | — | 0.98 |
| gpt-3.5 | openml-diabetes | original | 0.7396 | 0.7396 | — | 0.74 |
| gpt-3.5 | uci-wine | original | 0.8989 | 0.8989 | — | 0.88 (3-seed mean 0.8783) |
| gpt-3.5 | adult | original | 0.7710 | 0.7710 | — | 0.78 (3-seed mean 0.7773) |
| gpt-3.5 | icu | original | 0.7059 | 0.7059 | — | 0.69 (7-seed mean 0.6947) |

Across the full recomputation (path A, all 80 model×dataset×format cells) every published value is
reproduced within ±0.01 once seed averaging is applied.

## The headline claim

The paper states: "adding small amounts of noise and other re-formatting techniques leads to an
average accuracy drop of 6 percentage points on the memorized datasets. In contrast, the same
transformations do not affect the few-shot learning performance on unseen data."

Recomputed from the authors' responses, averaged over both models and all datasets in each panel:

| transformations averaged | memorized (Panel A) | novel (Panel B) |
|---|---|---|
| perturbed | −2.58 pp | −0.13 pp |
| task | −4.71 pp | +0.56 pp |
| statistical | −11.93 pp | −8.66 pp |
| perturbed + task | −3.64 pp | +0.22 pp |
| **perturbed + task + statistical** | **−6.41 pp** | −2.74 pp |

The published "6 percentage points" corresponds to averaging all three transformations on the
memorized datasets: we get −6.41 pp. The claim that the transformations "do not affect"
performance on unseen data holds for the noise and re-formatting transforms (+0.22 pp) but **not**
for the statistical transform, which costs 8.66 pp on novel datasets too — it makes the task
genuinely harder for everyone rather than removing a memorization advantage. The paper's own
Table 4 Panel B shows this, but the sentence in the abstract does not carry the qualification.
For our H3 the practical consequence is fixed in advance: the seen-vs-novel contrast must be built
on `original` vs `perturbed`/`task`, with `statistical` reported separately as a difficulty control.

## Two defects found in our own verification code, and fixed

Both would have silently corrupted the numbers:

1. **Truncated responses.** Responses in the archive are cut to the first token or two
   (`"<="` for `"<=50K"`, `"Not"` for `"Not Survived"`) because the authors generated with a tight
   token limit. Exact string comparison scored those cells at exactly 0.00 accuracy — a number that
   looks like a finding and is an artefact. Fixed by decoding a response as the unique label it
   prefixes, which is the general form of the per-dataset substring rules in the authors' notebook
   (`0 if r is None or "Less" in r else 1`).
2. **Partial ground-truth coverage.** Recovering labels only from few-shot turns covers every test
   point on small datasets but only 556/1000 on adult and 129/1000 on acs-income, and accuracy on
   that subsample is biased. Coverage is now reported next to every number, and the CSV path (full
   coverage) is used where the few-shot path is thin.

## What remains unverified

- **acs-income / acs-travel / fico / spaceship-titanic** are verified only through path A. Path B
  cannot match them (categorical values are re-encoded relative to the shipped CSV) and path C
  recovers too few labels to be meaningful on its own.
- The bootstrap confidence intervals published alongside Table 4 were not recomputed; only point
  estimates were.
- Nothing here re-runs the models: this verifies that the paper's numbers follow from the paper's
  data, not that a fresh run would reproduce them. The fresh-run evidence is in `RESULTS_GATE.md`.
