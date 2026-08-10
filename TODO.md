# Roadmap — Project 1: Memorization of tabular data in Russian-language LLMs

Coarse-grained task list. Each block is meant to be picked up in its own session:
read the linked artefacts, do the block, tick the boxes, commit. Detailed decisions
live in the documents referenced from each block, not here.

Status: **week 3-4 of the plan** (dataset collection done, adapted pipeline not yet
validated).

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

## Next: block A — validate the adapted pipeline (§8, the remaining gate)

The HF/Russian pipeline must reproduce the English result of the unmodified
pipeline before it may be used for any hypothesis. Until this is ticked, no H1-H4
number counts.

- [ ] Run the notebook on Kaggle with `Qwen/Qwen2.5-7B-Instruct`, group `canon`,
      English prompts only.
- [ ] Confirm the mock controls inside the notebook behave (10/10 and 0/10).
- [ ] Compare the iris/wine/diabetes row-completion counts against `RESULTS_GATE.md`
      and against Bordt's Table 3 expectation for open models: header test passes
      widely, row completion fires mainly on iris.
- [ ] If the signal is absent everywhere: stop and diagnose the adapter, not the
      models (chat template, `apply_chat_template` fallback, truncation).
- [ ] Record the outcome in `RESULTS_GATE.md` §5 and pin the model revision in
      `models.lock`.

## Block B — H1 and H1b: does the Western canon survive Russian adaptation

- [ ] Pin revisions for the base↔adapted pairs: Qwen2.5-7B ↔ T-lite,
      Mistral-Nemo ↔ Vikhr-Nemo, Qwen2.5-7B ↔ ruadapt-Qwen, plus Llama-3.1-8B.
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
