# Roadmap — Project 1: Memorization of tabular data in Russian-language LLMs

Coarse-grained task list. Each block is meant to be picked up in its own session:
read the linked artefacts, do the block, tick the boxes, commit. Detailed decisions
live in the documents referenced from each block, not here.

Status: **week 8 of the plan** — block A closed. Block B has the full four-test
battery on the Mistral-Nemo ↔ Vikhr-Nemo pair in the completion probe; the two
Qwen pairs and the Llama control are still to run. Block C session C1 has run
on that pair (2026-09-08): anchor fired, fresh control at zero, and the one
Russian dataset that fires (govdomains) does so through rows that are
near-duplicates of their own prompt — the near-duplicate half of §5's baseline
has to be operationalised in an amendment before any H2 verdict is stated. H4a
and H3 were re-scoped by `AMENDMENT_6` before any Russian dataset was measured.

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

**Decision taken 2026-08-11: amendment first, then a 12B diagnostic.**
→ `AMENDMENT_3_H1B_OUTCOMES.md` (secondary outcome, chat-template policy, and the
branch below committed in advance). Next action: run the notebook with the two 12B
models already set in it. ~2 h each on a T4x2.

- [x] Promote mean normalized Levenshtein to a secondary H1b outcome, with number
      format canonicalisation implemented, not just described. → `src/metrics.py`
- [x] Probe and record where each model's chat template puts the system prompt.
      Mistral-Nemo moves it to the last user turn; Qwen and Vikhr do not.
- [x] First 12B attempt (2026-08-12) died of CUDA OOM on load: `device_map="auto"`
      planned a two-card placement that the transformers 5 loader materialised in
      bfloat16, so 9 GB of nf4 weights tried to occupy 29 GB. Fixed — single-device
      placement, explicit float16, and the run now aborts if quantization was
      requested and not applied. A 1 GB preflight settles it before any large
      download. → `src/check_quantization.py`
- [x] **Run `mistralai/Mistral-Nemo-Instruct-2407` and
      `Vikhrmodels/Vikhr-Nemo-12B-Instruct-R-21-09-24`** on canon/raw/EN.
      Quantization confirmed (8.14 GB, nf4, one card). 15/20 cells each; five lost
      to OOM in both, the same five. -> `RESULTS_12B_DIAGNOSTIC.md`
- [x] Apply the branch of `AMENDMENT_3_H1B_OUTCOMES.md` §4: criterion met (Vikhr
      passes header on 2), so block B proceeds as preregistered — but the floor is
      **not** a scale effect. The 7B model shows the strongest signal of the three.
- [x] **Audit our protocol against the authors' own code**, not against memory.
      Five deviations found, all from library defaults where they chose otherwise
      for open models — including `chat_mode`, which changes the probe entirely.
      Corrected. -> `AMENDMENT_4_PROTOCOL_ALIGNMENT.md`
- [x] **Power analysis for H1b** (`src/power_h1b.py`): the observed effect needs
      210 queries per arm, and iris tops out at 142, so exhausting the dataset
      still gives only 64% power. Seeds cannot fix a dataset-bounded ceiling.
- [x] **Run the 2x2**: base and adapted × chat and completion prompting, plan
      `probe`, reference protocol. Completion extracts about four times more and
      is the primary probe. → `RESULTS_PROMPTING_PROBE.md`
- [x] The rest of the battery in the completion probe (`h1b_rest`: header, first
      token, feature; 1836 calls per model, 4 h 20 min for the pair on two T4s).
      Header passes on iris and diabetes for both models; feature is at the floor
      for both.
- [x] First token re-measured with the seed bound on the completion branch, so
      that the pair answers the same rows, and scored offline against
      data-derived baselines. Iris positive for both, base ahead on 18 vs 10
      discordant rows (McNemar p = 0.19); diabetes and adult negative for both.
      → `AMENDMENT_6` §1, `src/rescore_first_token.py`,
      `results/first_token_corrected.json`
- [ ] Write the block B results document from the result files by script
      (`src/report_run.py` plus the paired contrasts) once the pair is complete.

**The floor problem, for reference.** The gate found extractable memorization on iris
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
- [x] First-token baselines implemented offline (`src/rescore_first_token.py`):
      mode and previous-row decide, LR/GBT reported as the row-conditional bound
      (`AMENDMENT_6` §1).
- [ ] Feature-completion baselines: best of mode / LR / GBT (§5), offline over the
      counts. Needed before any feature verdict; the feature cells measured so far
      are at the floor for both models, so no verdict waits on it yet.

- [ ] The two Qwen pairs (Qwen2.5-7B ↔ T-lite, Qwen2.5-7B ↔ ruadapt-Qwen) and the
      Llama-3.1-8B control on `probe` + `h1b_rest` in the completion probe. Llama
      is gated: licence accepted and `HF_TOKEN` in the session before, not during.
- [ ] Apply the preregistered decision rules (binomial tests against the stated
      baselines, Holm within the H1 family).
- [ ] H1b is the contribution: paired base vs adapted comparison per dataset
      (McNemar on the same rows), Wilcoxon across datasets. Retained, attenuated,
      or amplified? On the one pair measured so far: iris retained with the base
      ahead on every test, nothing significant, everything else at the floor for
      both — the power ceiling of `src/power_h1b.py` is binding.

## Block C — H2: Russian datasets (next GPU session)

Defined in `AMENDMENT_6` §2, §3 and §7 before any Russian file was measured. The
session itself is `notebooks/session.json`.

- [x] Preflight on the six Russian files: all decode and prompt under UTF-8 mode,
      first-token digits build, every prompt fits the context. The runner refuses
      Cyrillic files under any other default encoding; the notebook sets UTF-8 mode.
- [x] Instrument check on the Russian plans: perfect memorizer 100% on every
      applicable cell, echo mock zero. → `results/validation/gateA_ru_probe_*`
- [x] Plans priced from a cost model fitted to the logged queries
      (`src/price_plan.py`): `ru_probe` ≈ 7 h per model on a T4 by the conservative
      estimate, `ru_probe_long` ≈ 4 h — two sessions, not one.
- [x] **Session C1** (2026-09-08): `ru_probe` — iris anchor, hflabs_city, govdomains,
      mos_zemelnye_uchastki, mos_torgovye_obekty, trudvsem (fresh control); header
      and row completion; `raw`; completion probe; Mistral-Nemo and Vikhr-Nemo in
      parallel, same seed. 1416 calls per model, 12/12 cells each, 7 h 25 min,
      quantization confirmed, instrument check 10/10 and 0/10. Iris anchor fired
      for both (51/142 and 32/142 — byte-identical to the August runs, all 142
      responses). Every count recomputed from the raw log by `src/rescore_calls.py`.
- [x] Fresh control at zero for both models on both tests (row 0/250, header
      fail; minimum detectable rate 0.6%). The validity gate holds for row
      completion and header on this pair.
- [ ] **Decide the near-duplicate rule before any H2 verdict** (`AMENDMENT_7`,
      pending). Under the letter of §5 govdomains is positive for both models
      (32/250 and 19/250 against a 0.6% duplicate rate) and so is
      mos_torgovye_obekty on a single match (1/250, p = 2.5e-7 against a zero
      baseline). `src/prefix_baseline.py` shows what those matches are: rows that
      lie within edit distance 0.1 of a row already in the prompt (29 of 32, 17 of
      19), reproduced by copying the template, advancing a counter, and filling
      the region name from world knowledge; the one mos_torgovye match is a
      sequential id. Bordt's test assumes independent rows; registry exports
      sorted by organisation break that assumption, and the duplicate rate does
      not see it. The preregistration names a "duplicate/near-duplicate base
      rate" and never operationalised the second half. Options in `LOG.md`
      2026-09-09; the rule goes into a dated amendment with a transparency note,
      since the numbers were seen first.
- [ ] Write the block C1 results document once the rule is fixed
      (`src/report_run.py`, `src/prefix_baseline.py`, `src/compare_pair.py`).
- [x] First token on the Russian datasets: the library's own pre-check (the
      first feature predicted from the preceding rows, which `first_token_test`
      refuses to run past) rejects mos_zemelnye_uchastki, mos_torgovye_obekty and
      trudvsem on every seed and cannot run on hflabs_city, govdomains and
      russian_retail (near-unique string first feature). Iris passes. **Not
      scheduled on any Russian file** — a consequence of the preregistered rule,
      recorded, not amended. → `src/precheck_first_token.py`,
      `results/first_token_precheck.json`
- [ ] **Session C2**: `ru_probe_long` (russian_retail rows, iris anchor). Whether
      govdomains goes on its secondary serialisations depends on the
      near-duplicate rule above; by the letter it is the only cell positive under
      `raw`.
- [ ] Secondary serialisations (`utf8_semicolon`, decimal comma where its text
      differs), cells positive under `raw` first.
- [ ] Strong form: the dataset is positive for a Russian-centric model and negative
      for every multilingual control — needs the Qwen pairs and Llama on the same
      plans.
- [ ] Every Russian zero reported with its minimum detectable rate, digits per row
      and the iris anchor of the same session (`AMENDMENT_5` §1–2).

## Block D — H4: prompt language

- [ ] H4a runs under chat prompting only — the completion probe has no instruction
      to translate (`AMENDMENT_6` §4). Every cell positive under either probe, plus
      the full canon, is re-run under Russian instructions.
- [ ] Number-format normalisation before string comparison (Ward's caveat) — without
      it a decimal-comma artefact masquerades as a language effect.
- [ ] McNemar over paired verdicts; Wilcoxon over per-cell rates.
- [ ] Exploratory arbiter: on `obfuscated` probes an instruction-driven gap should
      collapse.

## Block E — H3: few-shot inflation

- [ ] **Precondition** (`AMENDMENT_6` §5): H3 runs for a model only if blocks B
      and C give it at least three datasets with a positive verbatim verdict. The
      one pair measured so far has iris, plus diabetes by the header test only.
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
