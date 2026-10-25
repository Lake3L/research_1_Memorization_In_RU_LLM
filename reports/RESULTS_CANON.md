# Block B and the method control — the canon in the completion probe

The Western canon (iris, uci-wine, openml-diabetes, titanic-train,
adult-train, california-housing) under completion prompting and the reference
protocol (`AMENDMENT_4`), for every model of the study: `probe` (row
completion) and `h1b_rest` (header, first token, feature). Block B tests H1
and H1b on the base ↔ adapted pairs of `PREREGISTRATION.md` §3; OLMo-7B-hf is
the method control of `AMENDMENT_8` §2, the one model of this size with
published per-dataset verdicts (Bordt et al. Table 3). The Mistral-Nemo ↔
Vikhr-Nemo pair is in `RESULTS_PROMPTING_PROBE.md`; this document collects the
rest as it runs, and grows session by session.

---

## 1. Session O1 (2026-10-05)

OLMo-7B-hf `h1b_rest` on one card, its `probe` on the other, then
Qwen2.5-7B-Instruct `probe` on the card the OLMo probe freed. nf4 on a T4,
transformers 5.14.1, seed 42, the tokenizer round-trip passed for both
models, instrument check 10/10 and 0/10 in all three runs. Every cell
reproduces from its raw log (`src/rescore_calls.py`).

| run | calls | time | planning figure |
|---|---|---|---|
| OLMo-7B-hf, `h1b_rest` | 1,836 | 2 h 06 min | 4.3 h |
| OLMo-7B-hf, `probe` | 912 | 45 min | 1.5 h |
| Qwen2.5-7B-Instruct, `probe` | 912 | 56 min | 1.6 h |

Both models had never run in completion mode and were priced as the slower
12B family; their measured costs are now in `src/price_plan.py`.

## 2. Qwen2.5-7B-Instruct, row completion

Under `AMENDMENT_7` R1–R4 (`src/prefix_baseline.py`,
`results/prefix_baseline_probe_20261005T101710Z.json`):

| file | library | R1+R4 | p (R2) | witnesses reproduced | row-shaped | verdict |
|---|---|---|---|---|---|---|
| iris | **84/142** | **58/103** | 9 × 10⁻⁶⁹ | 186/247 | 99% | **positive** |
| uci-wine | 0/170 | 0/170 | 1 | 12/1,360 | 100% | negative |
| openml-diabetes | 0/250 | 0/250 | 1 | 6/687 | 77% | negative |
| titanic-train | 0/250 | 0/250 | 1 | 4/484 | 98% | negative |
| adult-train | 0/100 | 0/97 | 1 | 0/100 | 100% | negative |

The same single-dataset pattern as every model so far, and the strongest
iris of any: 59% of rows verbatim. On the 142 identical prompts:

| against | Qwen2.5-7B | other | Qwen only / other only | exact McNemar p |
|---|---|---|---|---|
| Mistral-Nemo-Instruct-2407 | 84 | 51 | 52 / 19 | 0.0001 |
| Vikhr-Nemo-12B | 84 | 32 | 58 / 6 | 9 × 10⁻¹² |
| YandexGPT-5-Lite-8B pretrain | 84 | 76 | 33 / 25 | 0.36 |

Its gate run in chat mode (block A, `RESULTS_GATE.md` §6) gave 13/50; the
completion probe extracts more than twice the rate (59% against 26%), the
direction `RESULTS_PROMPTING_PROBE.md` found on the Nemo pair. The H1b contrast with its
adaptations T-lite and ruadapt-Qwen needs their runs on the same prompts
(sessions B1 and B2).

## 3. OLMo-7B-hf: void, and why

Every OLMo cell is zero, and the instrument says why before any reading of
the counts: 0% of its row answers are row-shaped, mean normalised distance
0.90–0.99. All 912 row answers and the header answers begin with the same
sentence — "The first thing you need to do is to find a good and reliable
online casino" — whatever the table; the feature answers are runs of " . ".

OLMo-7B-hf's tokenizer appends its end-of-text token, `<|endoftext|>`, to
every text it encodes. A completion prompt ending in end-of-text tells a base
model that the document is over, and it begins another, from nothing. None of
the other five tokenizers of the study does this: on 36 logged prompts each,
Mistral-Nemo, Vikhr-Nemo, both YandexGPT models and Qwen2.5-7B end on the
token of the prompt's last characters, and the tokenizers of T-lite and
ruadapt-Qwen do not append it either. The backend now detects the behaviour when it loads a tokenizer
(`appends_eos`, recorded in the load report) and sends completion prompts
without that one token (`src/hf_llm.py` `encode_prompt`); on the five other
tokenizers it changes no token of any prompt, so no earlier number moves.
The FAIL_ADAPTER rule of block A is what this is for: a zero made of answers
that are not rows is not a negative. All 19 OLMo cells of O1 are void and are
rerun in session B1; the logs stay in `results/` as evidence.

## 4. Session B1 (2026-10-23)

Five runs on two T4s in the dispatch order of `notebooks/session.json`.
Settings as in O1. Every run is complete, the instrument check gave 10/10
and 0/10 in all five, and the tokenizer round-trip passed. OLMo's load
report shows `appends_eos: true`; T-lite and Qwen show `false`. All 25 row
and feature cells reproduce from their raw logs (`src/rescore_calls.py`).

| run | calls | wall time | planning figure |
|---|---|---|---|
| OLMo-7B-hf, `probe` | 912 | 51 min | 1.0 h |
| OLMo-7B-hf, `h1b_rest` | 1,836 | 2 h 08 min | 3.0 h |
| Qwen2.5-7B-Instruct, `h1b_rest` | 1,836 | 2 h 38 min | 3.4 h |
| T-lite-it-1.0, `h1b_rest` | 1,836 | 2 h 31 min | 3.4 h |
| T-lite-it-1.0, `probe` | 912 | 55 min | 1.2 h |

The longer card ran 5 h 09 min against the 6.8 h planned. By generation time
the five runs took 0.71–0.77 of their planning figures. `src/price_plan.py
--validate` now covers 34 runs, and the largest ratio is still 0.94. T-lite
costs what Qwen costs, so it stays in the qwen-7b family.

## 5. OLMo-7B-hf, the method control, against Bordt et al. Table 3

With the end-of-text token removed, OLMo answers in rows: 98–100% of its row
answers have the file's field count, against 0% in O1. Scored under
`AMENDMENT_7` (`results/prefix_baseline_probe_20261023T132159Z.json`):

| file | header (rows recovered) | Bordt, header | row, library | row, R1+R4 | p (R2) | Bordt, row |
|---|---|---|---|---|---|---|
| iris | pass (3) | ? | 15/142 | 9/103 | 4 × 10⁻⁴ | ✓ |
| uci-wine | fail | ✗ | 0/170 | 0/170 | 1 | ✗ |
| openml-diabetes | fail | ✗ | 0/250 | 0/250 | 1 | ✗ |
| titanic-train | fail | ? | 0/250 | 0/250 | 1 | ✗ |
| adult-train | fail | ✗ | 0/100 | 0/97 | 1 | ✗ |
| california-housing | fail | — | not in `probe` | | | — |

Key to Bordt's symbols:
- ✓ means evidence of memorization.
- ✗ means no evidence.
- ? means an ambiguous result.

They are read off the arXiv HTML of the paper, because the PDF text
extraction drops them. Table 3 has no California Housing column. Its
Spaceship Titanic and FICO columns are not in our canon.

The results agree. Bordt et al. give a definite verdict in eight cells, and
ours is the same in all eight. Their two ambiguous cells split: iris header
passes with three rows, and titanic header fails.

The two studies differ in method:
- **Verdicts.** Bordt et al. judge by eye from printed completions; ours are
  formal tests.
- **Code and checkpoint.** They ran `allenai/OLMo-7B` through their own code
  (`AMENDMENT_4` §1). We ran its transformers-native conversion, in nf4.

The method control therefore passes. On the one model of our size class with
published per-dataset verdicts, three parts of our method together reproduce
the published pattern: iris rows extractable, the rest of the canon not. The
three parts are the instrument, the completion protocol and the `AMENDMENT_7`
rules.

OLMo's iris is the weakest of the study: 15/142, against 32–95 for the other
seven models on the same 142 prompts (with ruadapt-Qwen, §7). Its other tests are in the table of §6:
- First token on iris is above the data baseline. Diabetes and adult are at
  baseline.
- Feature completion is zero on four files.
- Wine feature is inconclusive. Only 78 of 170 answers are in the feature's
  shape; in the rest OLMo closes a code block and writes a README
  (`## License`). This is the `FAIL_ADAPTER` floor, which
  `src/rescore_calls.py` now counts for every feature cell.

## 6. Qwen2.5-7B-Instruct ↔ T-lite-it-1.0 (H1b)

This is the first Qwen pair. All four tests ran on identical prompts: the same
seed and plan, with Qwen's probe from O1. Row completion was scored with both
logs together (`results/prefix_baseline_probe_20261023T162400Z.json`):

| file | Qwen2.5-7B | T-lite | R1+R4, Qwen | R1+R4, T-lite | Qwen only / T-lite only | exact McNemar p |
|---|---|---|---|---|---|---|
| iris | 84/142 | **95/142** | 58/103 | **67/103** | 12 / 21 | 0.16 |
| uci-wine, openml-diabetes, titanic-train, adult-train | 0 | 0 | 0 | 0 | — | — |

Over all 142 iris queries the count is 20 / 31, p = 0.16.

The secondary outcome of `AMENDMENT_3` §2 is the mean normalised distance of
the row answers:

| | iris | uci-wine | openml-diabetes | titanic-train | adult-train |
|---|---|---|---|---|---|
| Qwen2.5-7B | 0.080 | 0.426 | 0.524 | 0.524 | 0.469 |
| T-lite | 0.055 | 0.426 | 0.519 | 0.540 | 0.432 |

The other three tests, with OLMo alongside. First token is scored against the
data (`results/first_token_h1b_rest_20261023T160000Z.json`):

| test | file | Qwen2.5-7B | T-lite | OLMo-7B-hf |
|---|---|---|---|---|
| header | iris | pass (5 rows) | pass (18 rows) | pass (3 rows) |
| header | openml-diabetes | pass (2 rows) | fail | fail |
| header | adult-train | pass (1 row) | fail | fail |
| header | uci-wine, titanic-train, california-housing | fail | fail | fail |
| first token | iris (baseline 0.493, previous row) | 121/142, p = 3 × 10⁻¹⁹ | 131/142, p = 3 × 10⁻²⁸ | 95/142, p = 2 × 10⁻⁵ |
| first token | openml-diabetes (0.280, mode) | 59/250, p = 0.95 | 60/250, p = 0.93 | 34/250, p = 1 |
| first token | adult-train (0.220, mode) | 59/250, p = 0.29 | 58/250, p = 0.35 | 56/250, p = 0.46 |
| feature | the five files | 0 everywhere | wine 1/170, titanic 1/250, others 0 | 0; wine inconclusive (78/170 answered) |

The row-conditional bound is a sensitivity check only (`AMENDMENT_6` §1).
Against it, iris first token keeps Qwen (p = 0.003) and T-lite (p = 2 ×
10⁻⁷) and loses OLMo (p = 0.99). Every feature cell of Qwen and T-lite has
all its answers in the feature's shape, so their zeros are negatives.

**Reading.** T-lite reproduces iris at least as well as its base on every
test. The row difference favours T-lite and is not significant. The
distances agree to within 0.04 on every file, with no consistent direction.

The header test is one greedy completion per split, so a count of recovered
rows is not a rate. There, Qwen passes on the first rows of diabetes and
adult, and T-lite passes on neither.

On the primary outcome, then, this pair retains its base's memorization.
Neither attenuation nor amplification is shown. The Nemo pair went the other
way: after the same rules, Vikhr-Nemo held 24 of the 103 rows and
Mistral-Nemo 36 (McNemar p = 0.029, `RESULTS_EXPOSURE.md` §38). The direction
is therefore not common to the two adaptations. (The first version of this
paragraph, 2026-10-23, put that p beside the library counts, 32 and 51 of
142; their own McNemar p is 0.0034, `RESULTS_PROMPTING_PROBE.md` §2.)

These are one cell of H1b, not its test. The confirmatory test is the paired
Wilcoxon over datasets in the full design (four variants, three seeds). The second Qwen pair, ruadapt-Qwen, runs next on the same prompts.

## 7. Session B2 (2026-10-23): ruadapt-Qwen

ruadapt_qwen2.5_7B_ext_u48_instruct ran `probe` and then `h1b_rest` on one
card. Qwen2.5-7B-Instruct ran `exposure_2` on the other
(`RESULTS_EXPOSURE.md` Part VIII). Settings as in B1. ruadapt's tokenizer
extends Qwen's vocabulary: the round-trip passed, `appends_eos` is false,
and the instrument check gave 10/10 and 0/10. All ten row and feature cells
reproduce from their logs.

| run | calls | wall time | planning figure |
|---|---|---|---|
| ruadapt-Qwen, `probe` | 912 | 58 min | 1.2 h |
| ruadapt-Qwen, `h1b_rest` | 1,836 | 2 h 34 min | 3.4 h |
| Qwen2.5-7B-Instruct, `exposure_2` | 2,662 | 4 h 20 min | 5.7 h |

The longer card took 4 h 20 min against 5.7 h planned. All three runs came
in at 0.75 of their figures, and `--validate` over 37 runs still shows a
worst ratio of 0.94.

Qwen answered the iris anchor of `exposure_2` with the same 142 answers,
byte for byte, as in its O1 probe. The measurement is deterministic across
sessions.

**Row completion**, scored with Qwen's O1 probe on identical prompts
(`results/prefix_baseline_probe_20261023T200418Z.json`):

| file | Qwen2.5-7B | ruadapt-Qwen | R1+R4, Qwen | R1+R4, ruadapt | Qwen only / ruadapt only | exact McNemar p |
|---|---|---|---|---|---|---|
| iris | 84/142 | 76/142 | 58/103 | 50/103 | 16 / 8 | 0.15 |
| uci-wine, openml-diabetes, titanic-train, adult-train | 0 | 0 | 0 | 0 | — | — |

Over all 142 iris queries the count is 22 / 14, p = 0.24. The answers are
row-shaped at the following rates, all above the floor:

| file | row-shaped |
|---|---|
| iris | 100% |
| uci-wine | 78% |
| openml-diabetes | 74% |
| titanic-train | 98% |
| adult-train | 100% |

Mean normalised distance (`AMENDMENT_3` §2):

| | iris | uci-wine | openml-diabetes | titanic-train | adult-train |
|---|---|---|---|---|---|
| Qwen2.5-7B | 0.080 | 0.426 | 0.524 | 0.524 | 0.469 |
| ruadapt-Qwen | 0.076 | 0.450 | 0.530 | 0.542 | 0.456 |

**The other three tests** (first token in
`results/first_token_h1b_rest_20261023T210223Z.json`):

| test | file | Qwen2.5-7B | ruadapt-Qwen |
|---|---|---|---|
| header | iris | pass (5 rows) | pass (23 rows) |
| header | openml-diabetes | pass (2 rows) | pass (1 row) |
| header | adult-train | pass (1 row) | fail |
| header | uci-wine, titanic-train, california-housing | fail | fail |
| first token | iris (baseline 0.493) | 121/142, p = 3 × 10⁻¹⁹ | 119/142, p = 1 × 10⁻¹⁷ |
| first token | openml-diabetes (0.280) | 59/250, p = 0.95 | 54/250, p = 0.99 |
| first token | adult-train (0.220) | 59/250, p = 0.29 | 50/250, p = 0.80 |
| feature | the five files | 0 everywhere | wine 1/170, others 0; every answer in shape |

On iris first token, ruadapt also keeps the row-conditional bound
(p = 0.01). Its 23 recovered header rows are the most of any model on iris;
the next are YandexGPT pretrain with 20, Mistral-Nemo with 19 and T-lite
with 18.

**Reading.** ruadapt-Qwen holds iris slightly below its base. The row
difference is 8 queries of 103 and is not significant (p = 0.15). The
distances move by at most 0.03. The iris first-token count moves by two
queries, and the two files at baseline stay at baseline. This pair, too,
retains on the primary outcome.

## 8. H1b over the three pairs so far

Iris row completion is the one canon file where any model reproduces rows.
It was scored after `AMENDMENT_7` R1 and R4 on the same 103 kept queries for
every model:

| pair | base | adapted | base only / adapted only | exact McNemar p | reading |
|---|---|---|---|---|---|
| Mistral-Nemo ↔ Vikhr-Nemo | 36/103 | 24/103 | 19 / 7 | 0.029 | attenuated |
| Qwen2.5-7B ↔ T-lite | 58/103 | 67/103 | 12 / 21 | 0.16 | retained |
| Qwen2.5-7B ↔ ruadapt-Qwen | 58/103 | 50/103 | 16 / 8 | 0.15 | retained |

Sources: `results/prefix_baseline_ru_probe_20260908T121529Z.json` (C1's iris
anchor), `results/prefix_baseline_probe_20261023T162400Z.json` and
`results/prefix_baseline_probe_20261023T200418Z.json`.

Only one of the three Russian adaptations loses a detectable part of what its
base holds, the 12B Vikhr-Nemo. The two Qwen adaptations stay within chance
of their base, in opposite directions: T-lite above, ruadapt below. The
other four files of `probe` are zero for all five models, so they contribute
no paired information. H1b's confirmatory Wilcoxon over datasets therefore has
one informative dataset per pair in this design.

What is established is narrower and still useful: Russian adaptation does
not, as a rule, erase verbatim memorization of the canon. Whether it
attenuates it depends on the adaptation.

The two Qwen adaptations do differ from each other. Session B3 repeated both
iris anchors byte for byte, on the same 103 kept queries. There T-lite has
26 rows that ruadapt lacks and ruadapt 9 that T-lite lacks, McNemar
p = 0.006 (`results/prefix_baseline_exposure_2_20261024T212338Z.json`). This
contrast is exploratory, since H1b compares each adaptation with its base.
Each stays within chance of Qwen, but on opposite sides, and the two sides
are far enough apart to be told apart.

## 9. Files

Session O1:
- `results/calls_probe_Qwen_Qwen2.5-7B-Instruct_raw_en_completion_20261005T101710Z.jsonl`
  and its `gateA_` file.
- `results/prefix_baseline_probe_20261005T101710Z.json`.
- Void, evidence only:
  `results/calls_{probe,h1b_rest}_allenai_OLMo-7B-hf_raw_en_completion_20261005T092707Z.jsonl`
  and their `gateA_` files.

Session B1:
- `results/calls_probe_allenai_OLMo-7B-hf_raw_en_completion_20261023T132159Z.jsonl`.
- `results/calls_h1b_rest_allenai_OLMo-7B-hf_raw_en_completion_20261023T141400Z.jsonl`.
- `results/calls_h1b_rest_Qwen_Qwen2.5-7B-Instruct_raw_en_completion_20261023T132159Z.jsonl`.
- `results/calls_h1b_rest_t-tech_T-lite-it-1.0_raw_en_completion_20261023T160000Z.jsonl`.
- `results/calls_probe_t-tech_T-lite-it-1.0_raw_en_completion_20261023T162400Z.jsonl`.
- Each of the five call logs above has its `gateA_` file.
- `results/prefix_baseline_probe_20261023T132159Z.json` (OLMo).
- `results/prefix_baseline_probe_20261023T162400Z.json` (Qwen and T-lite,
  paired).
- `results/first_token_h1b_rest_20261023T160000Z.json`.

Session B2:
- `results/calls_probe_RefalMachine_ruadapt_qwen2.5_7B_ext_u48_instruct_raw_en_completion_20261023T200418Z.jsonl`.
- `results/calls_h1b_rest_RefalMachine_ruadapt_qwen2.5_7B_ext_u48_instruct_raw_en_completion_20261023T210223Z.jsonl`.
- Both call logs have their `gateA_` files.
- `results/prefix_baseline_probe_20261023T200418Z.json` (Qwen and ruadapt,
  paired).
- `results/first_token_h1b_rest_20261023T210223Z.json`.
- Qwen's `exposure_2` run is listed in `RESULTS_EXPOSURE.md` §48.

```
python src/prefix_baseline.py results/calls_probe_Qwen_Qwen2.5-7B-Instruct_raw_en_completion_20261005T101710Z.jsonl results/calls_probe_RefalMachine_ruadapt_qwen2.5_7B_ext_u48_instruct_raw_en_completion_20261023T200418Z.jsonl --group canon --out results/prefix_baseline_probe_20261023T200418Z.json
python src/rescore_first_token.py results/calls_h1b_rest_Qwen_Qwen2.5-7B-Instruct_raw_en_completion_20261023T132159Z.jsonl results/calls_h1b_rest_RefalMachine_ruadapt_qwen2.5_7B_ext_u48_instruct_raw_en_completion_20261023T210223Z.jsonl --out results/first_token_h1b_rest_20261023T210223Z.json
```

Session O1 and B1:

```
python src/rescore_calls.py results/calls_probe_Qwen_Qwen2.5-7B-Instruct_raw_en_completion_20261005T101710Z.jsonl --results results/gateA_probe_Qwen_Qwen2.5-7B-Instruct_raw_en_completion_20261005T101710Z.json
python src/prefix_baseline.py results/calls_probe_Qwen_Qwen2.5-7B-Instruct_raw_en_completion_20261005T101710Z.jsonl --out results/prefix_baseline_probe_20261005T101710Z.json
python src/rescore_calls.py results/calls_h1b_rest_allenai_OLMo-7B-hf_raw_en_completion_20261023T141400Z.jsonl --results results/gateA_h1b_rest_allenai_OLMo-7B-hf_raw_en_completion_20261023T141400Z.json
python src/prefix_baseline.py results/calls_probe_allenai_OLMo-7B-hf_raw_en_completion_20261023T132159Z.jsonl --group canon --out results/prefix_baseline_probe_20261023T132159Z.json
python src/prefix_baseline.py results/calls_probe_Qwen_Qwen2.5-7B-Instruct_raw_en_completion_20261005T101710Z.jsonl results/calls_probe_t-tech_T-lite-it-1.0_raw_en_completion_20261023T162400Z.jsonl --group canon --out results/prefix_baseline_probe_20261023T162400Z.json
python src/rescore_first_token.py results/calls_h1b_rest_Qwen_Qwen2.5-7B-Instruct_raw_en_completion_20261023T132159Z.jsonl results/calls_h1b_rest_t-tech_T-lite-it-1.0_raw_en_completion_20261023T160000Z.jsonl results/calls_h1b_rest_allenai_OLMo-7B-hf_raw_en_completion_20261023T141400Z.jsonl --out results/first_token_h1b_rest_20261023T160000Z.json
python src/price_plan.py --validate
```
On Windows: the project `.venv` interpreter and `PYTHONUTF8=1`. Every call
log is rescored the same way as the first, each with its own `gateA_` file.
