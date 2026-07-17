# Preregistration: Memorization of Tabular Data in Russian-Language LLMs

**Date:** 2026-07-17 (committed before any model was queried).
**Status:** frozen. Changes after this commit are amendments: dated, committed separately, and listed in the paper's Deviations section. Negative results will be published regardless of outcome.

## 1. Research question

Have Russian-centric and Russian-adapted LLMs memorized (a) the Western canon of tabular datasets and (b) open Russian-language tabular datasets — and how does this contamination distort the evaluation of their few-shot abilities, in particular when the evaluation is conducted in Russian?

## 2. Instruments

Primary: the four verbatim-memorization tests of Bordt et al. (COLM 2024, arXiv:2404.06209) as implemented in `tabmemcheck` v0.1.6, adapted to local HF models and Russian prompt scaffolding (adaptation changes prompts/backends only, never test logic or success criteria). Knowledge-level heuristics (feature names / values / distributions) are run and reported **separately** from verbatim tests, following the paper's Table 2 taxonomy.

Secondary (confound control): MCQ contamination probes with `like` and `obfuscated` dataset variants following Silvestri et al. (arXiv:2510.20351); normalized-Levenshtein near-match rate as an approximate-memorization metric (Ward et al., arXiv:2512.08875, with number-format normalization before comparison).

Note correcting our own working plan: `tabmemcheck` uses **no logprobs** (query access only, per its README). Open weights are required for determinism, cost, and base-vs-adapted comparisons — not for logprobs. API models can in principle run all tests but are excluded from confirmatory analyses (version drift, no greedy-decoding guarantee).

## 3. Models

Pinned HF revisions (commit hashes) will be frozen in `models.lock` before the first run; substitutions are allowed only before the first run of a given model and must be logged in LOG.md.

- **Russian-centric (confirmatory):** T-lite (Qwen2.5-7B-based), Vikhr (Nemo-12B or 7B variant), ruadapt-Qwen 7B-class.
- **Multilingual controls (confirmatory):** the corresponding base instruct models, forming explicit base↔adapted pairs: Qwen2.5-7B-Instruct ↔ T-lite; Mistral-Nemo-Instruct ↔ Vikhr-Nemo; Qwen2.5-7B-Instruct ↔ ruadapt-Qwen. Plus Llama-3.1-8B-Instruct as an off-family control.
- **Optional (exploratory only):** T-pro (32B-class, if rented compute allows), GigaChat/YandexGPT APIs.

Decoding: greedy (temperature 0) for all memorization tests and few-shot classification; temperature 0.7 only for distribution-sampling heuristics (as in Bordt). 4-bit quantization allowed; the same quantization config is used for both members of every base↔adapted pair.

## 4. Datasets

1. **Western canon (6):** Adult, Kaggle Titanic, California Housing, Iris, Wine, OpenML Diabetes — byte-exact canonical CSVs shipped with `tabmemcheck` (memorization tests are run on raw CSV rows, never translated or reformatted).
2. **Russian pre-cutoff (target ≥4, minimum 2):** open Russian-language tabular datasets published as CSV before 2023-01-01. Inclusion criteria (frozen now): Russian headers and/or categorical values; ≥300 rows; at least one high-entropy feature usable for the feature-completion test; evidence of public availability before 2023 (archive links). Candidate sources: Kaggle RU competitions, data.gov.ru / data.mos.ru archives, Rosstat, HF datasets. The concrete list, canonical file canonicalization (encoding UTF-8, separator, row order) and SHA-256 hashes will be frozen in a dated amendment **before any model sees any of these files**. If fewer than 2 qualifying datasets are found by end of week 3, H2 is reduced to a case study (per plan §2.6) — stated here in advance.
3. **Fresh controls (≥2):** collected by us after 2026-06-01 (after all model cutoffs), including one construction-matched "twin" of a Russian pre-cutoff dataset (mirroring Bordt's Adult↔ACS Income pairing). Not sourced from Kaggle mirrors. Collection dates and hashes committed.

Fresh controls are a validity gate: a positive verbatim verdict on a fresh dataset for any model marks that model×test combination as unreliable and excludes it from confirmatory analysis (reported in the paper).

## 5. Test-level decision criteria (replacing Bordt's manual verdicts)

For each (model × dataset × prompt-language):

- **Header test:** pass = verbatim continuation of at least one full next CSV row, best of splits in rows 2/4/6/8 (Bordt's criterion, unchanged).
- **Row completion (N=250 or dataset-bounded):** positive if exact-match rate exceeds the dataset's duplicate/near-duplicate base rate, one-sided exact binomial test.
- **Feature completion (N=250):** positive if exact-match rate on the designated high-entropy feature exceeds the conditional-baseline rate (best of mode / LR / GBT predicting that feature), one-sided binomial.
- **First token test (N=250):** positive if first-token accuracy exceeds baseline (best of mode / LR / GBT), one-sided binomial; run only where the row-independence pre-check passes.

**Verbatim verdict** for a model×dataset = positive if the header test passes or ≥1 of the three count tests is significant after correction. Per-dataset entropy diagnostics (duplicate-row share, digits-per-row) are always reported alongside (Ward covariate).

Multiple-comparison correction: Holm within each hypothesis family (H1, H2, H4 families = all model×dataset×test p-values entering that hypothesis; H3 family = its contrasts), α=0.05.

## 6. Hypotheses and decision rules

**H1 — Russian-centric models have memorized the Western canon.** English prompts, canonical CSVs.
*Confirmed* if ≥1 Russian-centric model gets a positive verbatim verdict on ≥2 of 6 canon datasets. *Refuted* if no Russian-centric model gets a positive verdict on any canon dataset while the paired base models do (i.e., adaptation erased an existing signal — reported as such); if base models also show nothing, the positive-control gate (§8) has failed and no claim is made.
*H1b (confirmatory, direction unspecified):* within each base↔adapted pair, the difference in memorization signal (row-completion rate; first-token accuracy) — does Russian adaptation retain, attenuate, or amplify inherited memorization? Paired comparison per dataset, Wilcoxon across datasets.

**H2 — Russian-centric models have memorized Russian pre-cutoff datasets.**
*Confirmed* if ≥1 (Russian-centric model × Russian pre-cutoff dataset) receives a positive verbatim verdict (any prompt language). *Strong form:* the same dataset is negative for all multilingual controls. *Refuted* if all such combinations are negative **and** the positive-control gate passed (tests demonstrably work in our hands) **and** fresh controls stayed negative.

**H3 — Memorization inflates few-shot performance.** Bordt Section-4 design: formats original/perturbed/task/statistical, 20-shot stratified, temperature 0, LR and GBT baselines, majority-class lift and Cohen's κ reported per dataset (Gorla control), identical serialization across models, 3 seeds for few-shot example selection (mean ± std).
Primary contrast (difference-in-differences): Δ = mean over *seen* datasets of (acc_original − acc_task) minus the same mean over *fresh* datasets, per model.
*Confirmed* if Δ > 0 with p < 0.05 (permutation test over datasets, per model, Holm over models) and point estimate ≥3 p.p. *Refuted* if the 95% CI upper bound < 3 p.p. Power analysis targeting the 6 p.p. effect reported by Bordt will be run **before** main H3 runs; if power < 80%, we add datasets/seeds or narrow the model set, documented as an amendment before unblinding results.
"Seen" status is assigned by our own H1/H2 verdicts, not by publication date alone.

**H4 — Memorization-test outcomes depend on prompt language.** Same model, same byte-exact CSV; only instruction scaffolding (system prompts, few-shot wrapper templates, few-shot example files) switches EN↔RU. Run on all model×dataset cells with a positive EN verdict, plus the full canon for the quantitative comparison. String comparison uses number-format normalization so that decimal-separator artifacts cannot masquerade as a language effect.
*Confirmed* if (a) McNemar over paired positive/negative verdicts across cells shows significant flips (α=0.05), or (b) the paired per-cell relative change in quantitative signal (row-completion exact-match rate; first-token accuracy) is significant by Wilcoxon with median relative drop ≥25%. *Refuted* if neither test is significant and the median relative change is <10%. Arbiter for mechanism: on `obfuscated` probes (language-neutral content), an RU–EN gap driven by instruction semantics should collapse; this sub-analysis is exploratory.
*H4b (secondary):* same EN↔RU manipulation for few-shot classification accuracy on seen datasets (the wrapper is the language-bearing component most likely to matter).

## 7. Confound plan: verbatim memorization vs. textual-description knowledge

We commit to never claiming "memorization" from knowledge-level signals alone. Three instruments: (1) two-tier reporting per Bordt's taxonomy (knowledge heuristics vs. verbatim tests); (2) `like`/`obfuscated` MCQ probes per Silvestri on Russian datasets — beating random on `real` while not beating marginal-likelihood baselines or `like` ⇒ classified as statistical knowledge, not memorization; (3) Gorla's leakage taxonomy (complete overlap / label exposure / task leakage) used to classify every positive finding in the paper.

## 8. Positive-control gate (week 2, before any confirmatory run)

Reproduce Bordt's published results with unmodified `tabmemcheck`: header test passes on pre-2021 canon datasets and row-completion behavior matches Table 5 qualitatively for at least one strong public model; exact-number comparison against Tables 5/6 where the same model version is accessible. If reproduction fails, stop and diagnose (plan §2.7); no H1–H4 runs before the gate passes. The adapted (HF/RU) pipeline must additionally reproduce the EN results of the unmodified pipeline on one model×dataset before use.

## 9. Reproducibility commitments

Fixed global seed plus 3 seeds for anything sampled; pinned `requirements.txt`; pinned HF revisions in `models.lock`; canonical dataset files with SHA-256 hashes and collection dates; every LLM call logged (prompt, response, config) to `results/`; all paper numbers regenerable by one command from a clean clone. Compute: no local GPU — runs on Colab/rented single GPU; this bounds confirmatory models to ≤14B in 4-bit (32B-class exploratory only).

## 10. Stopping criteria (from working plan §2.7, restated as commitments)

- End of week 2: Bordt reproduction fails → stop and diagnose, do not proceed.
- End of week 4: adapted tests give no meaningful signal on positive controls → stop.
- End of week 6: no significant effect in any direction and no confidence the method works → wrap up as a short negative-result report; the negative result is published either way.
