# Block C, exposure sessions — Russian classifiers fire, and why

Two sessions on the Mistral-Nemo ↔ Vikhr-Nemo pair over the nine tables that
`AMENDMENT_9` froze for their public exposure.

**What they establish.** Under the preregistered rules two Russian
classifiers, МКБ-10 and ОКВЭД 2, are positive by row completion for both
models (E1), and H2e is confirmed by its letter. The test that `AMENDMENT_9`
and Part I named as decisive for reading that positive — feature completion,
the official name asked for by its code — is negative on both classifiers for
both models against its preregistered conditional baseline (E2). The rows the
models do reproduce are the ones whose name is a close variation of names
already in the prompt: the match rate falls from 29% to 2% as the next name
moves away from them. The positive is reconstruction of an inherently
predictable sequence (Prashanth et al., arXiv:2406.17746), not recall of the
reference table. The one table whose content the models do produce, ОКСМ, is
produced as world knowledge: 48% of full country names, and none of the 25
that carry the portal's own spelling. Parts I and II give the sessions,
Part III the family verdict and the reading.

---

# Part I — session E1

Session E1, plan `exposure_1`, 2026-09-20. Mistral-Nemo-Instruct-2407 and its
Russian adaptation Vikhr-Nemo-12B, one model per accelerator, 4-bit nf4 on a
T4, completion prompting, reference protocol, seed 42, published bytes
(`raw`). 1,416 calls per model; 11 of 11 cells; 6 h 38 min and 7 h 27 min,
against the 6.0 h conservative estimate. Quantization confirmed on both
(8.14 GB, `sdpa`, one device). Instrument check inside the session: perfect
memorizer 10/10, format echo 0/10.

Every count below reproduces from the raw call log by an independent path
(`src/rescore_calls.py`: 6 of 6 cells per model, exactly), and every verdict
is by `AMENDMENT_7` R1–R4 with the witness tiers of `AMENDMENT_9` §3
(`src/prefix_baseline.py`).

---

## 1. What fired

| dataset | group | base: library | base: R1+R4 | p (R2) | witness | adapted: library | adapted: R1+R4 | p (R2) | witness | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| iris (anchor) | canon | 51/142 | 36/103 | 9 × 10⁻³⁴ | 151/247 | 32/142 | 24/103 | 2 × 10⁻¹⁸ | 130/247 | positive, both |
| mkb10_v2 | A | 24/250 | 16/223 | 1 × 10⁻²⁴ | 28/248 | 15/250 | 9/223 | 3 × 10⁻¹² | 20/248 | **positive, both** |
| okved2 | A | 21/250 | 16/240 | 3 × 10⁻²⁴ | 28/228 | 13/250 | 9/240 | 5 × 10⁻¹² | 17/228 | **positive, both** |
| cardio_train | B | 0/250 | 0/250 | 1 | 4/461 | 0/250 | 0/250 | 1 | 5/461 | negative |
| alice_train_sessions | B | 0/250 | 0/250 | 1 | 0/738 | 0/250 | 0/250 | 1 | 0/738 | negative |
| mos_metro_stations_2022 | C | 0/250 | 0/245 | 1 | 0/250 | 0/250 | 0/245 | 1 | 23/250 | **inconclusive, §3** |

Header test: passes on iris only (19 rows recovered for the base, 5 for the
adaptation); fails on every other file, for both models.

**H2e is confirmed.** Two Russian pre-cutoff tables are positive for a
Russian-centric model under the preregistered rule, and the same tables are
positive for the multilingual base, so the strong form of §2 does not hold.
This is the first positive Russian cell in the study; session C1 on the
`AMENDMENT_1` files was negative everywhere.

The matches are not prefix continuation. The model-free copy-and-increment
predictor of R2 hits **zero** windows inside either cell, so no match is one
of its hits; the R2 null is 1.0 × 10⁻³ on both files (rule of three on
okved2, predictor on mkb10_v2). Excluding near-duplicate queries costs 27 of
250 queries on mkb10_v2 and 10 of 250 on okved2, and the count stays
significant at every τ of the curve down to 0.05 (mkb10_v2 base 22/240 at
τ = 0.05, 16/223 at 0.10, 4/167 at 0.20; okved2 18/246, 16/240, 5/207).

The matches are spread through the file, not clustered at its head: matched
rows on mkb10_v2 run from 855 to 14,638 of 14,772 (median 6,356), on okved2
from 121 to 2,573 of 2,986 (median 1,608). One match of 21 on okved2 and none
on mkb10_v2 lie in the first 5% of the file. Whatever the models saw, it was
not only the first rows that a notebook prints.

Examples, base model, exactly as logged (the eighth prompt row, then the true
ninth row, then the answer):

```
…2383,0408E834,E83.4,Нарушения обмена магния,2378,,0,,2015-01-23,,Актуальный
 true 2384,0408E835,E83.5,Нарушения обмена кальция,2378,,0,,2015-01-23,,Актуальный
  got 2384,0408E835,E83.5,Нарушения обмена кальция,2378,,0,,2015-01-23,,Актуальный

…1305,D,35.21.21,D35.21.21,Сжижение антрацита,,,2014-01-31,,0,,Актуальный,2014-02-01
 true 1306,D,35.21.22,D35.21.22,Сжижение каменного угля за исключением антрацита,…
  got 1306,D,35.21.22,D35.21.22,Сжижение каменного угля за исключением антрацита,…
```

## 2. What the positive means, and what it does not

The witness is the content tier in every case: the official `Наименование` of
a code that is absent from the prompt, reproduced verbatim as a field, 28
times of 248 available on mkb10_v2 and 28 of 228 on okved2 for the base model
(20 and 17 for the adaptation). No file-tier witness exists on these two
files, because a classifier has none: `AMENDMENT_9` §3 designated the tiers in
advance for exactly this case.

So the claim these cells support is **the reference table's content is
memorized**, not **this portal's file was seen**. The portal's CSV shape — the
`Идентификатор` column, `Дата актуализации`, `Статус`, the BOM — is published
by one portal; the code-to-name content is reprinted by every medical and
business reference site in the Russian web. The header test failing on both
files is consistent with that reading and was predicted in `AMENDMENT_9` §2:
the model does not know this file's header, it knows what the file is about.

Under Gorla's taxonomy (`PREREGISTRATION.md` §7) this is knowledge-level
overlap of the content with the training corpus, demonstrated by a verbatim
test rather than by a knowledge heuristic. It is reported as such, in both
tiers, wherever it appears in the paper.

A second reading has to be excluded before the finding is stated in the paper,
and cannot be excluded by this session: a hierarchical classifier is partly
*inherently predictable* (Prashanth et al., arXiv:2406.17746). "Нарушения
обмена кальция" after "Нарушения обмена магния" is a plausible continuation
for a model that knows medicine and nothing of the table. Two cells of the
next session decide it: feature completion on the same files asks for the name
given the code and the rest of the row, where a predictor that merely knows
the domain scores at the mode baseline, and the LR/GBT baselines of §5 are
computed on the same data. The fresh twin of `AMENDMENT_7` R5 is not required
here (near-duplicate share 9.4% and 6.2%, both under 10%).

## 3. The Moscow metro cell produced no answer

`mos_metro_stations_2022` must not be read as a negative. The base model
returned the empty string to 249 of its 250 row queries (well-formed answer
rate 0%), and the adaptation returned prose or unrelated text (3.6%
well-formed). A cell where the model never produces something shaped like a
CSV row measures the model's ability to answer, not its memory — the
`FAIL_ADAPTER` case of `RESULTS_GATE.md` §6, at cell level. It is reported as
inconclusive, with its well-formed rate, and it enters no family.

The file is the one shape in this session that combines a semicolon separator,
every field quoted, a trailing separator at the end of each row, and two
header lines (English, then Russian). cardio_train is also semicolon-separated
and answered at 99% well-formed, so the separator alone is not the cause. The
streets file of session E2 has the same shape and is expected to behave the
same way; if it does, both Moscow files are reported inconclusive under `raw`,
and whether to run them in a registered secondary serialisation is a decision
for a dated amendment, because that text was never published.

The well-formed rate also qualifies the Alice zero: 36% for the base and 10%
for the adaptation, against 99% on cardio_train. The Alice rows are the widest
in the set (168 digits per row) and the models often stopped before completing
one. The cardio_train zero is the only clean negative of group B, and it is a
strong one: 99% of answers were well-formed rows, and none was right.

## 4. The exposure covariate does not predict, in the form we can measure it

`AMENDMENT_9` §2 predicted the row-completion rate to rise with exposure.
Measured over the six datasets of this session, it does not:

| dataset | GitHub, file name | GitHub, fragments | ru-Wikipedia 2025 | base rate |
|---|---|---|---|---|
| cardio_train | 1,824 / 207 | 8 / 0 | — | 0.000 |
| alice_train_sessions | 382 | 0 / 0 | — | 0.000 |
| mkb10_v2 | — | 8 / 1 | 257,778 | 0.096 |
| okved2 | — | 21 / 73 | 19,205 | 0.084 |
| mos_metro_stations_2022 | — | 417 / 918 | 161,095 | inconclusive |
| iris (anchor) | 131,840 | 26,048 / 26,688 | 6,771 | 0.359 |

The most-copied file in the set is negative and the two files with almost no
GitHub presence are positive. The covariate as defined counts copies **in code
repositories**, which is the right proxy for a dataset that circulates as a
file (iris, cardio) and the wrong one for a classifier, whose content is
reprinted as prose on sites GitHub does not index. The Wikipedia column
separates the two positives from the two negatives correctly, but it is a
proxy for the subject's prominence, not for the table's text.

Stated plainly: what these models reproduce is not the file that is copied
most often, but the content that the web repeats in prose. Prediction 4 is
therefore not supported in the form it was written, and the honest reading is
that the covariate needs a web-scale count before it can carry the claim.
The infini-gram counts over Dolma answer the same question for English only.

## 5. The predictions of `AMENDMENT_9` §2, checked

| prediction | outcome |
|---|---|
| 1. header fails on every A and C file | **held** for both models; and it failed on cardio_train and Alice too, which the prediction allowed to pass |
| 2. mkb10_v2 and okved2 positive for a from-scratch Russian model | **open** — those models have not run; but both files are positive already for the Nemo pair, which the prediction did not require |
| 2b. cardio_train positive for every model | **refuted** on this pair: 0/250 at a 99% well-formed rate |
| 3. the `AMENDMENT_1` files stay negative | untested here; unchanged from C1 |
| 4. rate monotone in exposure rank | **not supported** as measured (§4) |

The pair contrast repeats what block B found and C1 repeated: the base is
higher than the adaptation in every comparable cell (iris 51 vs 32, mkb10_v2
24 vs 15, okved2 21 vs 13; McNemar over the queries the rules keep, mkb10_v2
p = 0.016, okved2 p = 0.016, iris p = 0.029, all discordant in the base's
favour). Russian adaptation attenuates an inherited signal here as well, and
the signal it attenuates is a Russian one.

## 6. The baseline session E2 will be judged against, fixed before it runs

§5 scores feature completion against the conditional baseline, "best of mode /
LR / GBT predicting that feature". The runner records only the mode rate, so
`src/feature_baseline.py` computes the rest from the data alone, and it is
committed **before** the session it judges. Where the feature is a name with
thousands of values, logistic regression and gradient boosting are neither
computable nor meaningful, and one nearest neighbour over the same encoded
columns takes their place as the conditional predictor; the reported baseline
is the largest of whatever was computed, which can only make a positive verdict
harder. Five folds, shuffled, seed 42 (`data/feature_baselines.json`):

| dataset | feature | classes | mode | 1-NN | baseline | smallest detectable rate at 250 |
|---|---|---|---|---|---|---|
| mkb10_v2 | Наименование | 11,877 | 0.0147 | 0.0000 | 0.0147 | 3.2% |
| okved2 | Наименование | 2,705 | 0.0020 | **0.0338** | 0.0338 | 5.6% |
| oksm | Полное наименование по ОКСМ | 414 | 0.0066 | **0.0331** | 0.0331 | 5.6% |
| okpdtr | Наименование | 8,015 | 0.0009 | 0.0018 | 0.0018 | 1.2% |
| cardio_train | age | 8,076 | 0.0005 | 0.0002 | 0.0005 | 0.8% |
| telecom_churn | Total day minutes | 1,667 | 0.0024 | 0.0006 | 0.0024 | 1.2% |

The nearest-neighbour column is the reason to compute this in advance: on
okved2 and oksm a predictor that merely looks up a similar row gets the name
right about 3% of the time, seventeen times the mode rate. A feature-completion
count on those two files has to clear that, not the mode.

## 7. What Part I left open

Section 2 above named the reading to exclude — that a hierarchical classifier
is partly predictable from the rows the prompt shows — and the cells that would
decide it. Part II is that decision.

---

# Part II — session E2

Session E2, plan `exposure_2`, 2026-09-23. Same pair, same setup: 2,662 calls
per model, 16 of 16 cells, 5 h 20 min and 6 h 32 min against the 4.9 h
conservative estimate. Quantization confirmed (8.14 GB, `sdpa`, one device);
instrument check 10/10 and 0/10. The iris anchor returned 51/142 and 32/142,
byte-identical to E1, to C1 and to the August runs, for the fourth time.

Every cell reproduces from the raw log (`src/rescore_calls.py`, 11 of 11 per
model), after two defects of the rescoring script were found and fixed on
this session's data (§15).

## 8. Feature completion: the models cannot name a code

Feature completion shows one observation with every column but the target and
asks for the target. On a classifier the target is the official name and the
conditioning columns include the code, so this is the direct test of whether
the code-to-name content is held. Scored against the conditional baseline
fixed before the run (Part I §6, `data/feature_baselines.json`):

| file | base | adapted | baseline | p, base | p, adapted |
|---|---|---|---|---|---|
| mkb10_v2 | 0/250 | 0/250 | 0.0147 (mode) | 1 | 1 |
| okved2 | 12/250 | 7/250 | 0.0338 (1-NN) | 0.14 | 0.74 |
| okpdtr | 0/250 | 0/250 | 0.0018 (1-NN) | 1 | 1 |
| oksm | 120/250 | 96/250 | 0.0331 (1-NN) | 2 × 10⁻¹⁰⁶ | 5 × 10⁻⁷⁴ |
| cardio_train | 0/250 | 0/250 | 0.0005 (mode) | 1 | 1 |
| telecom_churn | 1/250 | 0/250 | 0.0024 (mode) | 0.45 | 1 |

On МКБ-10 neither model names a single code of 250, below even the rate of
always answering the commonest name. Asked for T38.4, the base model answers
«Ушиб, ушиб мягких тканей»; the name is «Отравление пероральными
контрацептивами». On ОКВЭД 2 the base names 12 of 250, where a predictor that
looks up the nearest other row reaches 3.4%; the difference is not
significant, and the adaptation is below it. No matched value occurs in the
few-shot examples of its prompt, on any file.

## 9. ОКСМ: world knowledge, and the check that shows it

ОКСМ is the one table whose target the models produce at scale. The amendment
anticipated why: the content is ISO 3166, in every library and encyclopaedia,
and a positive there is "an upper bound on what world knowledge alone
reproduces". Splitting the queries by what carries the name
(`src/classifier_diagnostics.py`):

| name | base | adapted |
|---|---|---|
| current record, plain name | 89/121 (74%) | 79/121 (65%) |
| historical record, plain name | 31/104 (30%) | 17/104 (16%) |
| name carrying the portal's own parenthetical, e.g. «Гибралтар(Брит.)» | **0/25** | **0/25** |

The portal writes dependent territories with a parenthetical of its own; the
models answer «Гибралтар», «Остров Рождества», «Ангилья» — the encyclopaedia
form. Current names beat historical ones two to one, which is how prominence
is distributed, not how a table is. The ОКСМ feature positive is reported as
world knowledge under Gorla's taxonomy (`PREREGISTRATION.md` §7); it is also
the study's demonstration that the feature test fires on world knowledge.

## 10. Row completion in E2

| file | base, R1+R4 | adapted, R1+R4 | well-formed answers | verdict |
|---|---|---|---|---|
| okpdtr | 1/238, p = 0.16 | 1/238, p = 0.16 | 96% / 96% | negative |
| telecom_churn | 0/250 | 0/250 | 98% / 90% | negative |
| oksm | 0/250 | 0/250 | 1% / 1% | inconclusive |
| mos_streets_omk_um_2022 | 0/216 | 0/216 | 0% / 9% | inconclusive |

ОКПДТР and telecom_churn are clean negatives: the models answered with rows,
and the rows were wrong. The other two files returned almost no row-shaped
answer at all.

## 11. Why three files return no row: the trailing delimiter

The metro file of E1, and ОКСМ and the streets file here, are the only three
of the nine whose rows end with the delimiter — the metro and streets exports
by the portal's format, ОКСМ because its last column is empty in 98% of its
records. They are also exactly the three whose well-formed answer rate is
below 10%; every other file answers at 10–99%.

| file | rows ending with the delimiter | well-formed, base / adapted |
|---|---|---|
| oksm | 98% | 1% / 1% |
| mos_metro_stations_2022 | 100% | 0% / 4% |
| mos_streets_omk_um_2022 | 100% | 0% / 9% |
| the other six files | 0% | 10–99% |

The prompt ends on that delimiter. The base model then returns the empty
string (213, 249 and 249 times of 250); the adaptation begins its answer with
another delimiter and a line break, and the library scores the first line
only. Reading every line of every answer does not rescue a signal: the true
next row appears on some line of 1 answer in 1,500, across the three files and
both models. These zeros are not hidden positives, but they are not negatives
either. All six cells are reported inconclusive under the FAIL_ADAPTER rule of
the block A gate (fewer than half the answers row-shaped; `RESULTS_GATE.md`
§6), applied cell by cell to negative cells. The finding generalises beyond
this study: under the library's first-line criterion, row completion is
undefined on a file whose rows end with the delimiter, as the header test is
on a file whose first row exceeds its window (`AMENDMENT_6` §3).

---

# Part III — the H2e family, and what the positive is

## 12. The family, with Holm

Thirty p-values — every model × file × test of the nine tables that returned
one — with Holm at α = 0.05 (`src/family_holm.py`,
`results/h2e_family_20260923T185214Z.json`). Inconclusive cells stay in the
family, which only makes the correction stricter for the others.

| file | test | base | adapted | verdict |
|---|---|---|---|---|
| mkb10_v2 | row | 16/223, p_Holm 3 × 10⁻²³ | 9/223, p_Holm 8 × 10⁻¹¹ | **positive, both** |
| okved2 | row | 16/240, p_Holm 9 × 10⁻²³ | 9/240, p_Holm 1 × 10⁻¹⁰ | **positive, both** |
| oksm | feature | 120/250, p_Holm 7 × 10⁻¹⁰⁵ | 96/250, p_Holm 2 × 10⁻⁷² | **positive, both** — world knowledge (§9) |
| mkb10_v2, okved2, okpdtr | feature | 0, 12, 0 of 250 | 0, 7, 0 of 250 | negative |
| okpdtr, telecom_churn, cardio_train | row | 1, 0, 0 | 1, 0, 0 | negative |
| cardio_train, telecom_churn | feature | 0, 1 of 250 | 0, 0 of 250 | negative |
| alice, metro, streets, oksm | row | — | — | inconclusive (at most 36% row-shaped) |

The header test failed on all nine files for both models. **H2e is confirmed
by its preregistered letter; its strong form is not**: the multilingual base
and its adaptation are positive on the same files.

## 13. What the row-completion positive is

If a model held МКБ-10 or ОКВЭД 2, it would name a code when shown the code.
It does not (§8). If it held the classifier as an ordered text, it would
continue the list wherever the list went. It does not either. Over the E1
queries the match rate depends on one thing: how far the true row's name is
from the closest name among the eight rows the prompt already shows
(normalised Levenshtein; bins fixed in `src/classifier_diagnostics.py`):

| distance of the next name to the closest prompt name | mkb10_v2, base | mkb10_v2, adapted | okved2, base | okved2, adapted |
|---|---|---|---|---|
| under 0.20 | 12/42 (29%) | 9/42 (21%) | 6/39 (15%) | 5/39 (13%) |
| 0.20 to 0.35 | 8/47 (17%) | 4/47 (9%) | 6/41 (15%) | 4/41 (10%) |
| 0.35 to 0.50 | 2/40 (5%) | 1/40 (3%) | 7/54 (13%) | 3/54 (6%) |
| 0.50 and more | 2/121 (1.7%) | 1/121 (0.8%) | 2/116 (1.7%) | 1/116 (0.9%) |

«Нарушения обмена кальция» after «Нарушения обмена магния»; «Множественные
переломы бедренной кости закрытые» after the same name without «закрытые».
Where the next name is new, the models almost never produce it, and the few
exceptions read as knowledge of the classification rather than of the file
(«Гидроцеле неуточненное» → «Сперматоцеле», the next rubric of N43).

The near-duplicate rule of `AMENDMENT_7` did its job at the level it is
defined on, the whole row, and removed 8 of the 24 base matches on МКБ-10. It
could not see this, because in a classifier export the whole-row distance is
dominated by identifiers and dates that change by a digit, while the field
that carries the content is copied almost whole. That is a limitation of the
rule, not a failure of its application, and it is stated as such: on a
hierarchically ordered reference table, a verbatim row-completion positive
that survives a row-level near-duplicate rule can still be sequence
reconstruction, and reading it needs a field-level check or, better, a content
test that does not show the neighbours. The decomposition above is
exploratory; its bins are committed before any other model runs on these
files and will be applied to them unchanged.

## 14. The predictions of `AMENDMENT_9` §2, after both sessions

| prediction | outcome on the Nemo pair |
|---|---|
| 1. header fails on every A and C file | **held**; it also failed on all three course files |
| 2. mkb10_v2 and okved2 positive for a from-scratch Russian model | **open**: not yet run. On the Nemo pair both are positive by the rule and negative by the content test |
| 2b. cardio_train positive for every model | **refuted**: 0/250 by row and by feature, at 99% well-formed |
| 3. the `AMENDMENT_1` files stay negative | untested here |
| 4. rate monotone in exposure rank | **not supported** (Part I §4) |

The question that remains is the one prediction 2 asks, sharpened by §13: does
a model trained from scratch on a Russian-heavy corpus — YandexGPT-5-Lite,
GigaChat-20B — name the codes the Nemo pair cannot? A yes on the feature test
of МКБ-10 or ОКВЭД 2, against the same baselines, would be memorization of
Russian reference content that the multilingual base and its adaptation lack.
A no makes the H2 null of this study a property of models at this scale, not
of the files chosen.

## 15. Instrument findings from these sessions

Each found on this data and fixed or recorded before the numbers above were
written:

- **The rescoring script read Latin column names only.** Its feature-prompt
  parser matched `[A-Za-z_]` names, so on every Russian file it identified no
  row. It now cuts the prompt at known column names.
- **The library cuts feature answers at the first blank line; the call log
  does not.** In completion mode feature completion goes through
  `ChatWrappedLLM(ends_with="\n\n")`, and the log is written below that
  wrapper. An answer that starts with a blank line is logged in full and scored
  empty. Vikhr-Nemo starts 3 of its okved2, 7 of its oksm and 1 of its
  telecom_churn answers with a blank line; all eleven are correct and all are
  lost to the cut, and the library's counts (7, 96 and 0) are the ones that
  stand. No verdict changes either way. The rescoring now applies the cut: all 37 feature cells in
  every log reproduce exactly, and the port of the library's answer parser
  agrees with the library on all 6,090 logged answers.
- **A trailing delimiter leaves row completion undefined** (§11).
- **`prefix_baseline.py` skipped a whole dataset group** through a hard-coded
  default; fixed in E1.

## 16. Files

Session E1: `results/calls_exposure_1_*_20260920T174835Z.jsonl`,
`results/gateA_exposure_1_*_20260920T174835Z.json`,
`results/prefix_baseline_exposure_1_20260920T174835Z.json`. Session E2: the
same with `exposure_2` and `20260923T185214Z`. Across both:
`results/classifier_diagnostics_exposure_20260923T185214Z.json` (§§8, 9, 11,
13) and `results/h2e_family_20260923T185214Z.json` (§12).

```
python src/report_run.py "results/gateA_exposure_2_*20260923T185214Z.json"
python src/rescore_calls.py results/calls_exposure_2_<model>_*.jsonl --results results/gateA_exposure_2_<model>_*.json
python src/prefix_baseline.py results/calls_exposure_2_*.jsonl --out results/prefix_baseline_exposure_2_20260923T185214Z.json
python src/classifier_diagnostics.py --e1 <E1 base> <E1 adapted> --e2 <E2 base> <E2 adapted> --out results/classifier_diagnostics_exposure_20260923T185214Z.json
python src/family_holm.py --prefix results/prefix_baseline_exposure_*.json --diagnostics results/classifier_diagnostics_exposure_*.json --results "results/gateA_exposure_*.json" --out results/h2e_family_20260923T185214Z.json
```
