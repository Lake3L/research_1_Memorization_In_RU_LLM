# Roadmap — Project 1: Memorization of tabular data in Russian-language LLMs

Coarse-grained task list. Each block is meant to be picked up in its own session:
read the linked artefacts, do the block, tick the boxes, commit. Detailed decisions
live in the documents referenced from each block, not here.

Status: **week 4 of the plan** — block A closed on 2026-08-11: the adapted pipeline
reproduces the English result and the §8 gate is fully passed. Block B is blocked on
one design decision (the floor problem, below), not on compute.

---

## Done

- [x] **Novelty protocol** (plan §1.1, all five steps) — verdict: niche free, with
      positioning constraints. → `NOVELTY_CHECK.md`
- [x] **Preregistration** committed before any experiment. → `PREREGISTRATION.md`
- [x] **Reproduction of Bordt et al.** — 14/15 comparable cells consistent with
      Tables 2/5/6; GPT-4-0613 (the paper's own checkpoint) reproduces iris row
      completion 24/25 vs 125/136 published. → `RESULTS_GATE.md`
- [x] **Table 4 verified** from the authors' released chatlogs by three independent
      paths; headline effect recomputes to −6.41 pp. → `RESULTS_TABLE4.md`
- [x] **Measurement code validated** in both directions before spending money
      (perfect-memorizer mock ≈100%, format-echo mock ≈0%). → `src/mock_llm.py`
- [x] **Dataset collection** — 12 datasets registered with provenance, hashes and
      serialisation variants. → `AMENDMENT_1_DATASETS.md`, `data/registry.json`
- [x] **Kaggle/Colab notebook** built from a reviewable .py source.
      → `notebooks/kaggle_memorization_run.ipynb`

---

## Closed: block A — the adapted pipeline is validated (§8, gate passed 2026-08-11)

The HF/Russian pipeline had to reproduce the English result of the unmodified
pipeline before it could be used for any hypothesis. It does, on iris, by three of
the four tests. The decision rule was written into `RESULTS_GATE.md` §6 before the
run and the verdict was printed by the runner, not chosen afterwards.

- [x] Pin the model revisions. → `models.lock`
- [x] Make the datasets rebuildable on a clean machine and hash-verified there.
      → `src/fetch_data.py` (12/12 restored byte-exact after deletion)
- [x] Test the canon on the *published* bytes, not on pandas round-trips of them
      (the `raw` variant; wine differs in 99.4% of rows otherwise).
- [x] Validate the instrument over the whole block A plan, both directions:
      perfect memorizer 20/20 cells at 100%, echo mock zero everywhere.
      → `results/validation/gateA_gate_hf_*`
- [x] Run the HF backend end to end (Qwen2.5-0.5B on CPU) — chat template, greedy
      decoding, per-call JSONL log. → `src/smoke_hf_header.py`
- [x] State the gate rule before seeing any number: PASS / FAIL_ADAPTER /
      FAIL_NO_SIGNAL, with the adapter case separated from the model case by the
      share of answers that even have the shape of a CSV row.
- [x] First Kaggle attempt (2026-08-11) stopped at the data step: five canon hashes
      did not match. Root cause was a Windows line-ending conversion of the freeze
      itself, not the run. Fixed and re-frozen. → `AMENDMENT_2_LINE_ENDINGS.md`
- [x] **Run the notebook on Kaggle with `Qwen/Qwen2.5-7B-Instruct`.** 599 calls,
      20/20 cells, 1 h 12 min, no errors.
- [x] Both artefacts returned and committed: counts and the full call log.
- [x] Every countable cell recomputed from the raw log by an independent path.
      → `src/rescore_calls.py`, 11/11 reproduce exactly.
- [x] **GATE PASSED** — iris row completion 13/50 (p=1.3e-11), iris header pass,
      iris first token 0.78 vs 0.36. → `RESULTS_GATE.md` §6

## Block B — H1 and H1b: does the Western canon survive Russian adaptation

**Decide first: the floor problem.** The gate found extractable memorization on iris
and nowhere else. `Qwen2.5-7B-Instruct` is the base of two of the three pairs, so on
five of six canon datasets H1b would be comparing zero against zero. This is a design
decision, it changes what the study can claim, and it belongs in an amendment before
any run — not in a results file afterwards. The options, none of them free:

- **(a) Run the pairs as preregistered and report the floor.** Cheapest and most
  honest; H1b then rests on iris plus whatever the adapted models add. A null on four
  of five datasets is publishable under §10 but is a weak contribution.
- **(b) Add a larger confirmatory model.** Vikhr-Nemo-12B ↔ Mistral-Nemo-12B is
  already in `models.lock` and is 12B rather than 7B; running that pair *first* tests
  whether the floor is a size effect before committing the rest of the compute.
- **(c) Move to the more sensitive instruments.** First token fired on iris where row
  completion was weakest relative to GPT-4, and the near-match rate separates iris
  (38%) from adult (4%) from the rest (0%) where exact-match counts are all zero.
  Making near-match a preregistered secondary outcome for H1b would give the paired
  comparison a graded quantity instead of a binary one. Requires an amendment.
- **(d) Reconsider the surface.** Titanic is the paper's strongest row-completion
  signal (194/250) and gave zero here; our copy is a mirror whose byte-identity with
  Kaggle's original is unverified (see `AMENDMENT_1_DATASETS.md`). Worth resolving
  before concluding anything about titanic specifically.

- [x] Pin revisions for the base↔adapted pairs: Qwen2.5-7B ↔ T-lite,
      Mistral-Nemo ↔ Vikhr-Nemo, Qwen2.5-7B ↔ ruadapt-Qwen, plus Llama-3.1-8B.
      → `models.lock`. Note: Llama-3.1-8B is gated (manual approval) and needs an
      accepted licence plus `HF_TOKEN` in the session — arrange before, not during.
- [ ] Implement the preregistered baselines before any verdict: best of mode / LR /
      GBT for feature completion and first token (§5). Offline, no GPU. The mode-only
      baseline currently in use differs from the published best-of by 36 points on
      adult first token.

- [ ] Run all four memorization tests × 6 canon datasets × 4 serialisation variants,
      English prompts, 3 seeds.
- [ ] Apply the preregistered decision rules (binomial tests against the stated
      baselines, Holm within the H1 family).
- [ ] H1b is the contribution: paired base vs adapted comparison per dataset,
      Wilcoxon across datasets. Retained, attenuated, or amplified?

## Block C — H2: Russian datasets

- [ ] Same battery on the five Russian pre-cutoff datasets, both prompt languages.
- [ ] Fresh control must come out at zero; any positive verdict there invalidates
      that model×test cell (preregistered validity gate).
- [ ] Strong form: the dataset is positive for a Russian-centric model and negative
      for every multilingual control.

## Block D — H4: prompt language

- [ ] Every cell that is positive under English prompts is re-run under Russian.
- [ ] Number-format normalisation before string comparison (Ward's caveat) — without
      it a decimal-comma artefact masquerades as a language effect.
- [ ] McNemar over paired verdicts; Wilcoxon over per-cell rates.
- [ ] Exploratory arbiter: on `obfuscated` probes an instruction-driven gap should
      collapse.

## Block E — H3: few-shot inflation

- [ ] **Power analysis first** (preregistered, and it is not a formality). The honest
      target effect is the difference-in-differences of ≈3.9 pp measured from the
      authors' data, not the 6 pp headline, and our confirmation threshold is 3 pp.
      If power < 80%, add datasets rather than seeds — the test permutes over
      datasets.
- [ ] Write transform configs per Russian dataset (renames and recodes in Russian,
      so the format manipulation does not confound with the language factor).
- [ ] Run original / perturbed / task on seen and fresh datasets, 20-shot,
      temperature 0, 3 seeds, LR and GBT baselines alongside.
- [ ] Report lift over majority and Cohen's κ per dataset, not raw accuracy.
- [ ] `statistical` format runs as a difficulty control and is reported separately.

## Block F — writing and release

- [ ] Rerun the novelty protocol (plan §5 requires it within a week of submission).
- [ ] Limitations: retired checkpoints, sample sizes, licence gaps on two datasets,
      the Titanic mirror caveat, the statistical-format finding.
- [ ] One command from a clean clone reproduces every number.
- [ ] Reproducibility statement, negative results included.
- [ ] arXiv endorsement — start early, it is not same-day (plan §4).

---

## Standing rules

- Nothing is measured before its instrument is validated (mocks, then the gate).
- Every reported number regenerates from raw logs by a committed script.
- Discrepancies are published, not smoothed: the iris 0.40 vs 0.26 gap is in the
  results file with its explanation.
- Dataset files are pinned by hash; a run against an unmatched hash is void.
- Negative results ship.
