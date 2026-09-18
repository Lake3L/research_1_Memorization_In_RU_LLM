# Amendment 6 to PREREGISTRATION.md — the first-token baseline, and the scope of block C

**Date:** 2026-09-07. **Scope:** the baseline against which the first-token test
is decided (§5); the serialisation design for the Russian datasets (§4); which
cells the header test can be run on; the form H4a takes under the completion
probe of `AMENDMENT_4`; and a precondition for H3. No confirmation threshold,
correction procedure, model or dataset changes. Everything here is stated before
any Russian dataset has been measured.

---

## 1. First-token test: scored offline, against predictors that see what the model sees

**The rule.** The first-token test is scored from the call log by
`src/rescore_first_token.py`, never from the line the library prints. The
decision baseline is the better of two predictors that use only the information
the model was given — the dataset's most common first token (*mode*) and the
first token of the last prefix row (*previous row*). The one-sided exact binomial
test is against that baseline, Holm within the family as before. The
row-conditional predictors that §5 names for this test — logistic regression and
gradient boosting fitted on the row's other features — are reported beside it as
a sensitivity bound and do not decide.

**Why the library's line is not a baseline.** `first_token_test` reports, under
the label "baseline", the number of *model responses* equal to the most common
first token. That is a property of the model's answers, not of the data, and it
differs between two models on the same file: on adult-train it reads 0.292 for
Mistral-Nemo and 0.076 for Vikhr-Nemo, where the data-derived value is 0.220 for
both. A baseline that moves with the model under test cannot decide whether the
model beat it. The library's completion branch also does not forward the seed to
the row sampler; the runner binds it for the duration of the call
(`seeded_row_completion` in `src/run_repro.py`) so that both members of a pair
answer the same rows and the comparison is paired. Test logic and success
criterion are untouched, as §2 requires.

**Why the row-conditional predictors do not decide.** The prompt of this test is
eight complete rows followed by nothing: the model continues the file. Logistic
regression and gradient boosting are given the target row's other features,
which are not in the prompt. A predictor that reads what the model cannot read
bounds how predictable the token is *from the row* — worth reporting, because it
says how much a model that had never seen the file could know if it were handed
the row — but it is not a baseline for the task the model was set. §5 wrote a
single baseline clause for two tests; for feature completion, whose prompt does
contain the row's other features, it stays exactly as written.

**Transparency.** This rule was written after the six cells below had been
scored under both families, and on iris the families disagree. The rule is
decided on the information argument, not on the outcome — which is why the
previous-row predictor, which *raises* the iris baseline from 0.408 to 0.493, is
part of it. The paired base-versus-adaptation contrast that H1b rests on
(McNemar over the same rows) uses no baseline at all and is unaffected by any
choice made here.

**The six cells, rescored** (`results/first_token_corrected.json`; 142 or 250
paired queries, same rows for both models):

| model | dataset | rate | mode | previous row | **p (decision)** | LR | GBT | p (row-cond.) | library's line |
|---|---|---|---|---|---|---|---|---|---|
| Mistral-Nemo | iris | 117/142 = 0.824 | 0.408 | 0.493 | **2.7e-16** | 0.753 | 0.713 | 0.029 | 0.415 |
| Vikhr-Nemo | iris | 109/142 = 0.768 | 0.408 | 0.493 | **1.8e-11** | 0.753 | 0.713 | 0.39 | 0.380 |
| Mistral-Nemo | openml-diabetes | 52/250 = 0.208 | 0.280 | 0.148 | 0.996 | 0.231 | 0.234 | 0.86 | 0.652 |
| Vikhr-Nemo | openml-diabetes | 55/250 = 0.220 | 0.280 | 0.148 | 0.987 | 0.231 | 0.234 | 0.73 | 0.680 |
| Mistral-Nemo | adult-train | 57/250 = 0.228 | 0.220 | 0.180 | 0.40 | 0.401 | 0.435 | 1.0 | 0.292 |
| Vikhr-Nemo | adult-train | 52/250 = 0.208 | 0.220 | 0.180 | 0.70 | 0.401 | 0.435 | 1.0 | 0.076 |

Iris is positive for both models under the decision baseline; diabetes and
adult are negative for both under every baseline. Paired on the same 142 iris
rows, the base model is right where the adaptation is wrong on 18 rows and the
reverse on 10 (McNemar p = 0.19): the same direction as row completion and the
header test, and not significant on its own. Under the library's line, Vikhr on
adult-train would have been reported positive.

---

## 2. Serialisation of the Russian datasets: encoding is not an experimental condition

`AMENDMENT_1` froze five serialisations per Russian dataset: the published bytes
(`raw`), UTF-8 with comma, UTF-8 with semicolon, cp1251 with semicolon and decimal
comma, and UTF-8 with semicolon and decimal comma. The instrument decodes a file
to text before any of it reaches a model, so what the model sees is the decoded
text and not the bytes. The cp1251 variant decodes to exactly the text of the
UTF-8 decimal-comma variant on every dataset where it exists, and where a file
carries no decimal numbers the two semicolon variants coincide as well:

| dataset | distinct decoded texts | which coincide |
|---|---|---|
| hflabs_city | 4 | cp1251 = UTF-8 decimal comma |
| govdomains | 4 | (no cp1251 variant registered) |
| mos_zemelnye_uchastki | 4 | cp1251 = UTF-8 decimal comma |
| mos_torgovye_obekty | 3 | UTF-8 semicolon = cp1251 = UTF-8 decimal comma |
| russian_retail | 4 | (no cp1251 variant registered) |
| trudvsem_vacancies_2026 | 2 | raw = UTF-8 comma; all semicolon forms identical |

**The rule.** Conditions are distinct decoded texts. The primary condition of
block C is `raw`, the published bytes, as for the canon. The secondary
conditions are `utf8_semicolon` and, where its text differs, `utf8_semicolon_decimal_comma`.
`cp1251_semicolon` is no longer a condition; the encoding of the published file
is recorded in the registry as provenance and is not manipulated. `utf8_comma`
is a pandas round trip of `raw` that differs from it only in quoting and number
formatting and is not run.

For H2, a dataset's verbatim verdict may come from any of its distinct
serialisations; every serialisation that is run enters the Holm family. Order:
`raw` on every Russian dataset first, then the secondary serialisations, cells
positive under `raw` first, the rest as compute allows.

---

## 3. Cells the header test cannot answer

The header test hands the model the first 500 characters of the file and passes
when the continuation reproduces at least one complete data row verbatim. On
`russian_retail` the header and the first row together are 1,713 characters, so
the 500-character window ends inside the first row and contains no row the test
could count; the perfect-memorizer mock fails the cell. The cell is **not
applicable**: it is not run and is reported as such, not as a fail. This holds
for any dataset whose header plus first row exceed the window; among the twelve
frozen datasets it is the only one.

Row completion on `russian_retail` is unaffected and stays at 250 queries. Its
rows are the longest in the study (median 359 tokens, against 20 for iris and
85–170 for the other Russian files), and 8% of them exceed 900 tokens and cannot
be reproduced inside the 1000-token generation budget; that ceiling is reported
beside the cell together with the digits-per-row covariate.

---

## 4. H4a under the completion probe

`AMENDMENT_4` made completion prompting the primary probe. Its prompt is rows of
the file and nothing else — no system prompt, no instruction, no few-shot
wrapper. There is no language-bearing component in it, so the manipulation H4a
describes has no primary-probe form.

**The rule.** H4a is tested under chat prompting, the secondary probe, on every
cell positive under either probe plus the full canon as §6 specifies. Its
conclusion is about instruction-driven extraction and is stated as that; it
does not carry over to the primary probe. H4b, the few-shot wrapper language, is
unchanged, and the number-format normalisation rule stays for both.

---

## 5. H3 has a precondition, stated before block C is measured

H3 contrasts datasets the model has memorized with fresh ones. For the two 12B
models the primary outcome has so far found one such dataset, iris; the header
test adds openml-diabetes. The preregistered power analysis permutes over
datasets and cannot reach 80% power over one or two.

**The rule.** H3 runs for a model only if blocks B and C together give that
model at least three datasets with a positive verbatim verdict. Otherwise H3 is
reported for that model as not testable, with the seen set that was found, and
no few-shot compute is spent on it. This is a precondition, not an outcome;
where H3 runs it remains confirmatory as preregistered.

---

## 6. A reproducibility commitment (§9): text decoding

`tabmemcheck` decodes files with the interpreter's default encoding. All runs
are made in UTF-8 mode (`PYTHONUTF8=1`, set by the notebook for every run), and
the runner refuses to start when a selected file contains non-ASCII bytes and
the default encoding is not UTF-8. A results file for the Russian datasets
therefore cannot come from a run that decoded them differently.

---

## 7. What block C measures first

Plan `ru_probe` (`src/run_repro.py`): iris as the in-session anchor of
`AMENDMENT_5` §2, the four Russian pre-cutoff datasets with rows under 200 tokens,
and the fresh control — header and row completion, `raw`, completion prompting,
reference protocol, 250 queries per row cell (142 on iris), the same seed for
both models of the pair. Plan `ru_probe_long` follows with `russian_retail`. First
token and feature completion on the Russian datasets, and the secondary
serialisations, come after these two, with the cells they are worth running on
chosen from what the two plans find. All of block C is on the Mistral-Nemo ↔
Vikhr-Nemo pair first; the two Qwen pairs and the Llama control follow on the
same plans.
