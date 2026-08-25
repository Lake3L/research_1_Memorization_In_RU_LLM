# The prompting probe — how the question is asked changes the answer by a factor of four

**Date:** 2026-08-25. **Question:** `AMENDMENT_4` §3 asked whether completion-mode
prompting extracts more than chat-mode prompting, and whether the base-vs-adapted
difference survives once the chat-template confound is removed. Both are answered.

## 1. What ran

Canon in its published bytes, `raw` variant, English, seed 42, reference protocol
(five few-shot blocks, eight prefix rows), 4-bit on a Kaggle T4. Row completion on
five datasets at the query counts of `AMENDMENT_4` §5, which exhaust iris and wine.

Three of the four planned runs completed:

| run | model | prompting | status |
|---|---|---|---|
| 1 | Mistral-Nemo-Instruct-2407 (base) | completion | complete |
| 2 | Vikhr-Nemo-12B (adapted) | completion | complete |
| 3 | Mistral-Nemo-Instruct-2407 (base) | chat | complete |
| 4 | Vikhr-Nemo-12B (adapted) | chat | **not run** — session ended |

Both models loaded quantized (8.14 GB, nf4, SDPA attention, one device) and the
mock controls behaved in every session: perfect memorizer 10/10, format echo 0/10.

## 2. Iris row completion, dataset exhausted (n = 142)

| model | prompting | exact | rate | 95% CI |
|---|---|---|---|---|
| Mistral-Nemo (base) | chat | 13/142 | 0.092 | [0.050, 0.151] |
| Mistral-Nemo (base) | **completion** | **51/142** | **0.359** | [0.280, 0.444] |
| Vikhr-Nemo (adapted) | **completion** | **32/142** | **0.225** | [0.160, 0.303] |
| Vikhr-Nemo (adapted) | chat | — | — | not run |

### Finding 1 — the prompting mode is worth a factor of four

Same model, same data, same seed, same protocol. Asking through an instruction
wrapper: 0.092. Asking the model to continue the text: 0.359.

Fisher exact **p = 7.3 × 10⁻⁸**, odds ratio 5.6. The comparison is unpaired
because chat mode consumes the random stream differently and samples different
rows.

Every earlier measurement in this project used chat prompting. On this evidence
they understated extractable memorization by roughly four times, and the floor
reported in `RESULTS_12B_DIAGNOSTIC.md` was in part a property of the question
rather than of the models.

### Finding 2 — the adaptation has *less* extractable memorization than its base

In completion mode both models received byte-identical prompts in identical order
— verified from the logs — so this is a paired comparison over the same 138 rows,
and McNemar applies.

| | count |
|---|---|
| both models reproduced the row | 22 |
| **base only** | **29** |
| **adapted only** | **10** |
| neither | 77 |

**McNemar exact, 39 discordant pairs: p = 0.0034.** Base 0.370, adapted 0.232.

Completion mode uses no chat template and no system prompt, so the confound
recorded in `AMENDMENT_3` §3 — the two models place the system prompt differently
— does not exist in this arm. The comparison is of weights, not of instruction
formats.

**This reverses the direction seen under chat prompting.** The 12B pilot had the
adapted model ahead (7/50 against 4/50); here, asked directly, the base model is
ahead and significantly so. The most economical reading is that the chat-mode
comparison was partly measuring which model follows an autocomplete instruction
better, which is what `AMENDMENT_3` §3 predicted it might be. The fourth run will
test that reading rather than leaving it as an inference.

## 3. Everything else is still zero, now at proper sample sizes

| dataset | n | exact | min detectable rate | digits/row | well-formed answers |
|---|---|---|---|---|---|
| uci-wine | 170 | 0/170 | 0.9% | 36.1 | 99-100% |
| openml-diabetes | 250 | 0/250 | 0.6% | 19.1 | 74-100% |
| titanic-train | 250 | 0/250 | 0.6% | 18.3 | 93-99% |
| adult-train | 100 | 0/100 | 3.0% | 16.0 | 92-100% |

Identical in both prompting modes and both models. These are strong nulls, not
absences of measurement: at these sample sizes a rate above roughly one percent
would have been detected. The answers are well-formed CSV rows almost throughout,
so the zeros are content failures rather than format failures.

Titanic remains the sharpest disagreement with the literature — Bordt et al.
report 194/250 for GPT-3.5 — and the standing caveat applies: our copy is a public
mirror whose byte-identity with the Kaggle original is unverified, and these tests
are byte-sensitive.

## 4. Against the criterion written before the run

`AMENDMENT_4` §6 said completion mode becomes the primary probe if it "lifts
row-completion rates materially above chat mode", and illustrated that with "any
dataset moving from ≤0.05 to ≥0.20".

The illustration is not satisfied: iris was already at 0.092 under chat prompting,
above the 0.05 the parenthetical named, and no other dataset moved off zero. The
parenthetical did not anticipate that the only dataset carrying signal would
already sit above that line.

The condition itself is satisfied decisively: 0.092 → 0.359 on the same rows of
the same file, p = 7.3 × 10⁻⁸. **Completion mode becomes the primary probe for H1
and H1b**, and the chat-template confound is thereby designed out of the primary
comparison rather than controlled for.

## 5. Verification

Every reported count was recomputed from the raw call logs by
`src/rescore_calls.py`, which reconstructs the ground truth from the prompts and
the frozen CSVs without reading the run's own result file. Where the log is
complete the counts reproduce exactly: iris 51/142, wine 0/170, diabetes 0/250 for
run 1; iris 32 of the 138 logged calls for run 2, against 32/142 reported, so the
four unlogged calls contained no further match; iris 12 of 135 logged against
13/142 reported.

Two defects were found in the course of this and both are fixed:

- **The call logs are prefixes.** They were downloaded from the running session
  while the runs were still appending to them, so they stop at the moment of
  download. The result files are written at the end of a run and are complete.
  The notebook now copies each run's artefacts out as soon as that run finishes,
  so what is downloaded is a static copy.
- **Our re-scoring matched prompts by their last row.** On a dataset with
  duplicate rows that lookup lands on the first occurrence and takes the wrong row
  as ground truth. Iris carries 2% duplicates, which was enough to move the
  re-scored count by one. Matching is now on the whole prefix block, and with that
  correction the re-score agrees exactly.

## 6. What remains

1. **The fourth cell** — the adapted model under chat prompting. Without it the
   reversal in §2 is an inference across two protocols rather than a measured
   interaction. It is queued as the next session.
2. **Complete logs** for the three runs already made, if the Kaggle session still
   holds them; otherwise a repeat of the two completion runs, which carry the
   headline result.
3. H1 and H1b now run in completion mode, at the query counts of `AMENDMENT_4` §5.
