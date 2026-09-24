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
- [x] **The near-duplicate rule** → `AMENDMENT_7_NEAR_DUPLICATE_RULE.md`
      (2026-09-09): near-duplicate queries excluded with a τ-curve, the null
      never ε (duplicate / prefix-predictor / rule of three), witness columns
      per dataset (`data/witness_columns.json`), fragment queries excluded, the
      fresh twin a precondition for registry-style cells. Scored by
      `src/prefix_baseline.py`; runner and `detectability.py` use the same null;
      mocks re-validated. **C1 under the rule: iris positive for both, every
      Russian cell negative, H2 not confirmed on the Nemo pair.**
- [x] **Model set widened** → `AMENDMENT_8_MODEL_SET_AND_EXPEDITIONS.md`:
      YandexGPT-5-Lite-8B pretrain and instruct, GigaChat-20B-A3B base and
      instruct (both from scratch, Russian-centric, enter H1 and H2, not H1b),
      OLMo-7B-hf as the published-verdict and open-corpus control. Pinned in
      `models.lock`. Hypothesis stated before any run: the H2 null is exposure,
      not adaptation.
- [ ] **E1, duplication covariate**: public copies of two distinctive rows per
      dataset (GitHub code search, web, infini-gram over Dolma v1.7). First
      probe done: iris rows 310 and 212 hits in Dolma, Russian rows 0 (Dolma is
      English-heavy; the GitHub/web count is the one that matters). GitHub code
      search works through the authenticated `gh` CLI (2026-09-18): iris row
      47,104 files, govdomains row 2; the fragment rule (which fields, how many)
      is fixed in the next amendment before any count enters a table.
- [x] **High-exposure tables frozen** → `amendments/AMENDMENT_9_HIGH_EXPOSURE_TABLES.md`
      (2026-09-18): nine files in registry group `ru_exposure` with hashes
      (`src/prepare_exposure_datasets.py`), witness columns in two tiers
      (file / content), the exposure covariate measured by one rule for all
      21 datasets (`src/exposure_counts.py` → `data/exposure_counts.json`),
      hypothesis H2e and its predictions written before any run, plans
      `exposure_1` (6.0 h) and `exposure_2` (4.9 h) priced and validated on
      both mocks. Re-fetch sources verified against the frozen hashes for the
      six re-fetchable files; the Moscow releases and the Kaggle competition
      file travel with the run.
- [x] **Session E1** (2026-09-20): `exposure_1` on the Nemo pair, 1416 calls per
      model, 11/11 cells, 6 h 38 m and 7 h 27 m, quantization confirmed, mocks
      10/10 and 0/10, every cell reproduced from the raw log. **МКБ-10 and
      ОКВЭД 2 are positive for both models under `AMENDMENT_7` — the first
      positive Russian cells in the study; H2e is confirmed.** cardio_train is a
      clean negative at a 99% well-formed rate; the Moscow metro cell produced
      no well-formed answer at all and is inconclusive, not negative. The
      exposure covariate did not predict the outcome in the form it was
      measured. → `reports/RESULTS_EXPOSURE.md`
- [x] **Session E2** (2026-09-23): `exposure_2` on the Nemo pair, 16/16 cells,
      every cell reproduced from the raw log. **Feature completion is negative
      on the classifiers** (МКБ-10 0/250 both; ОКВЭД 12 and 7 of 250 against a
      3.4% lookup baseline): the models cannot name a code. The E1 row positive
      is sequence reconstruction — the match rate falls from 29% to 2% as the
      next name moves away from the prompt's names. ОКСМ feature is positive as
      world knowledge (0/25 portal-specific names). Rows ending with the
      delimiter leave row completion undefined (metro, streets, ОКСМ:
      inconclusive). H2e family with Holm: confirmed by its letter, strong form
      not. → `reports/RESULTS_EXPOSURE.md` Parts II–III
- [x] Read the YandexGPT licence in full (2026-09-24, LOG.md): research use
      free and unrestricted for this study; name the model with its copyright
      notice in the paper.
- [ ] **Session Y1** (`notebooks/session.json`): the YandexGPT-5-Lite-8B pair,
      `exposure_2` then `probe` on each card (6.4 h estimated). The decisive
      question after E2: does a from-scratch Russian model name МКБ-10 and
      ОКВЭД codes by feature completion, against the same baselines? Score with
      `src/prefix_baseline.py`, `src/classifier_diagnostics.py` (bins frozen)
      and `src/family_holm.py`.
- [ ] Then: `exposure_1` and `ru_probe` on the YandexGPT pair; C2
      `ru_probe_long` on the Nemo pair; OLMo; GigaChat after its smoke test.
- [ ] Then both plans on the YandexGPT and GigaChat pairs and on the controls,
      so that the exposure figure has every model on the same files. The strong
      form of H2e is already ruled out on these two files: the multilingual base
      is positive on them too, so they cannot distinguish a Russian-centric
      model by presence alone — only by rate.
- [ ] **If the streets cell also returns no well-formed answer**: both Moscow
      files are reported inconclusive under `raw`, and running them in a
      registered secondary serialisation needs a dated amendment (that text was
      never published; `AMENDMENT_6` §2).
- [ ] **E2, corpus composition** from the technical reports and model cards
      of every model (delegated 2026-09-09, verify personally): table in the
      block C results document.
- [ ] **E3, the fresh registry twin** (`AMENDMENT_5` §4, `AMENDMENT_7` R5):
      candidates found 2026-09-09 (LOG.md): FNS register of disqualified persons
      (CSV, weekly, sequential record number, per-record dates, but personal
      data of individuals), FNS unified SME register (XML, monthly,
      `ДатаВклМСП`, legal entities), and a data.mos.ru set via a free API key,
      to be checked from a Russian address. Author chooses; register source,
      schema, collection date and hash before any model runs on it.
- [ ] Read the GigaChat custom code before the first Kaggle preflight; GigaChat
      smoke test at full size on a T4 before pricing any plan. (The YandexGPT
      licence is read, 2026-09-24.)
- [ ] Write the block C1 results document (`src/report_run.py`,
      `src/prefix_baseline.py`, `src/compare_pair.py`, E2 table).
- [x] First token on the Russian datasets: the library's own pre-check (the
      first feature predicted from the preceding rows, which `first_token_test`
      refuses to run past) rejects mos_zemelnye_uchastki, mos_torgovye_obekty and
      trudvsem on every seed and cannot run on hflabs_city, govdomains and
      russian_retail (near-unique string first feature). Iris passes. **Not
      scheduled on any Russian file** — a consequence of the preregistered rule,
      recorded, not amended. → `src/precheck_first_token.py`,
      `results/first_token_precheck.json`
- [ ] **Sessions, in the order of `AMENDMENT_8` §5**, with E1 and E2 slotted
      first on each pair and every session planned at ≤ 10 h by the
      conservative estimate (`AMENDMENT_9` §7): C2 `ru_probe_long` on the
      Nemo pair; YandexGPT pretrain + instruct on `probe` and `ru_probe`;
      OLMo-7B-hf on the canon (`probe`, `h1b_rest`) with the cell-by-cell
      comparison to Bordt Table 3; GigaChat base + instruct after the smoke
      test; then the Qwen pairs and Llama-3.1-8B (block B); the twin last, on
      every model that ran on the file it mirrors. Secondary serialisations of
      govdomains are no longer queued: the cell is negative under `raw`.
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
