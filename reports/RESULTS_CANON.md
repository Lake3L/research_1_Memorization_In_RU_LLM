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

## 4. Files

Session O1: `results/calls_probe_Qwen_Qwen2.5-7B-Instruct_raw_en_completion_20261005T101710Z.jsonl`
and its `gateA_` file, `results/prefix_baseline_probe_20261005T101710Z.json`;
void, evidence only: `results/calls_{probe,h1b_rest}_allenai_OLMo-7B-hf_raw_en_completion_20261005T092707Z.jsonl`
and their `gateA_` files.

```
python src/rescore_calls.py results/calls_probe_Qwen_Qwen2.5-7B-Instruct_raw_en_completion_20261005T101710Z.jsonl --results results/gateA_probe_Qwen_Qwen2.5-7B-Instruct_raw_en_completion_20261005T101710Z.json
python src/prefix_baseline.py results/calls_probe_Qwen_Qwen2.5-7B-Instruct_raw_en_completion_20261005T101710Z.jsonl --out results/prefix_baseline_probe_20261005T101710Z.json
```
On Windows: the project `.venv` interpreter and `PYTHONUTF8=1`.
