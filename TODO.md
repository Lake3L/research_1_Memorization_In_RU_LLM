# Roadmap — Project 1: Memorization of tabular data in Russian-language LLMs

Coarse-grained task list. Each block is meant to be picked up in its own session:
read the linked artefacts, do the block, tick the boxes, commit. Detailed decisions
live in the documents referenced from each block, not here.

Status: **week 3-4 of the plan** — dataset collection done; the adapted pipeline is
built, instrument-validated against mocks over the whole block A plan, and waiting
on one GPU session to be validated against a real model.

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

Preparation is done and the decision rule is written down in `RESULTS_GATE.md` §6
before the run. What is left is the run itself.

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
- [ ] **Run the notebook on Kaggle with `Qwen/Qwen2.5-7B-Instruct`, group `canon`,
      variant `raw`, English prompts only.** Expect 1.5-3 h on a T4 in 4-bit.
- [ ] Bring back both artefacts: `results/gateA_*.json` and `results/calls_*.jsonl`.
      The call log matters more than the counts — it is what lets a scoring rule be
      revised without paying for the session again.
- [ ] Fill `RESULTS_GATE.md` §6 from the run. If FAIL_ADAPTER: diagnose the chat
      template and truncation, not the model. If FAIL_NO_SIGNAL: that is a result
      about a 7B model's extractability and it is reported, not tuned away (§10).

## Block B — H1 and H1b: does the Western canon survive Russian adaptation

- [x] Pin revisions for the base↔adapted pairs: Qwen2.5-7B ↔ T-lite,
      Mistral-Nemo ↔ Vikhr-Nemo, Qwen2.5-7B ↔ ruadapt-Qwen, plus Llama-3.1-8B.
      → `models.lock`. Note: Llama-3.1-8B is gated (manual approval) and needs an
      accepted licence plus `HF_TOKEN` in the session — arrange before, not during.
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
