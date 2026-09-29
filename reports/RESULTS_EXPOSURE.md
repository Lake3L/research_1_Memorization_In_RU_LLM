# Block C, exposure sessions — Russian classifiers fire, and why

Two sessions on the Mistral-Nemo ↔ Vikhr-Nemo pair over the nine tables that
`AMENDMENT_9` froze for their public exposure (Parts I–III), and the first
session on a model trained from scratch on a Russian-heavy corpus, the
YandexGPT-5-Lite-8B pair (Part IV).

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

**What Part IV adds (2026-09-28).** The YandexGPT-5-Lite-8B pair names ОКВЭД 2
codes: 32 and 28 of 250 official names from their codes against a 3.4%
lookup baseline (p = 1.6 × 10⁻¹⁰ and 4.1 × 10⁻⁸), where the Nemo pair
reached 12 and 7. It is the first cell in the study where a Russian
classifier's content is produced from its code at a rate the conditional
baseline does not explain, and the first half of `AMENDMENT_9`'s prediction
2. МКБ-10 is not named (10 of 250 by the instruct model, raw p = 0.004,
not expected to survive the family's correction). The same pair holds iris
more strongly than any model so far (76/142 and 53/142), and its pretrain
member beats its instruct member on the same rows (McNemar p = 0.008). Two
instrument findings: a pretrain model may not answer the feature prompt at
all — it opens a new record instead — so the FAIL_ADAPTER floor now applies
to feature cells, which re-labels two Vikhr-Nemo cells of Part II from
negative to inconclusive without changing what they say; and the trailing
delimiter of §11 leaves row completion undefined only for a model that stops
at it — the Yandex pair answers those files.

**What Part V adds (2026-09-29).** With `exposure_1` the H2e family of the
YandexGPT pair is complete. It is positive on МКБ-10 by row completion and on
ОКВЭД 2 by row and feature completion, for both members, after Holm over the
pair's 30 p-values and over all four models' 60. On МКБ-10 the frozen distance
diagnostic of §13 now separates the two lineages on identical prompts: where
the next rubric's name shares nothing with the names the prompt shows, the
Yandex models name it 18 times of 121 and the Nemo models 2 and 3 times
(paired McNemar p = 3 × 10⁻⁵ and 6 × 10⁻⁵). The Yandex pair holds the
classification as ordered text, which the Nemo pair only reconstructs. It does
not hold the portal's file: no model reproduces a parent id or record-code
prefix that the prompt does not show (0 of 11 and 0 of 2 for every model), and
code-to-name retrieval on МКБ-10 stays at 10 of 250. Of `AMENDMENT_9`'s
prediction 2 both files are now met for a from-scratch Russian model;
prediction 2b, cardio positive for every model, is refuted on all four.

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

*Correction, 2026-09-28 (Part IV §20).* The adaptation's answers on МКБ-10 and
ОКПДТР begin with a blank line 173 and 213 times of 250, and the library's
wrapper scores an answer that begins with a blank line as empty (§15). Only
77 and 37 of its 250 answers on these two files were therefore read, below
the FAIL_ADAPTER floor, and the two cells are **inconclusive**, not negative
(§12, updated). Read without the cut, its 250 and 246 answers still name
**0** codes, so the sentence above stands as a description of the answers;
the formal verdict is the one that changes. The base model's cells (245–250
answers read) are unaffected.

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

*Narrowed, 2026-09-28 (Part IV §21).* The YandexGPT pair answers the same two
E2 files with a line break and a row, 82–95% well-formed, and those cells are
conclusive for it. The criterion is undefined for a model that treats the
trailing delimiter as the end of the record — the Nemo pair — not for the
file as such; the six Nemo cells stay inconclusive.

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
| okved2 | feature | 12/250 | 7/250 | negative |
| mkb10_v2, okpdtr | feature | 0, 0 of 250 (98%, 100% read) | 0, 0 of 250 (31%, 15% read) | base negative; adapted **inconclusive** (corrected 2026-09-28, §8, Part IV §20) |
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

---

# Part IV — session Y1: the YandexGPT-5-Lite-8B pair

## 17. The session

Session Y1, plans `exposure_2` and then `probe`, 2026-09-27 20:33 to
2026-09-28 02:14 UTC, the third attempt (the first two, 2026-09-24/25, are void;
§22). `yandex/YandexGPT-5-Lite-8B-pretrain` and `-instruct`, revisions
`f4aec3ab…` and `b5568117…`, one per card, 4-bit nf4 (5.6 GB, `sdpa`, one
device), completion prompting, reference protocol, seed 42, published bytes.
The runner installed transformers 5.14.1 before importing it (tokenizers
0.22.2, sentencepiece 0.2.1, torch 2.10.0); the versions are in every results
file, and the decoding defect of the void attempts is gone — iris rows come back
byte-exact and Cyrillic names match verbatim. Instrument check 10/10 and 0/10
in all four runs.

| run | calls | cells | time | estimate |
|---|---|---|---|---|
| pretrain, `exposure_2` | 2,662 | 16/16 | 4 h 33 min | 4.6 h |
| instruct, `exposure_2` | 2,662 | 16/16 | 3 h 42 min | 4.6 h |
| pretrain, `probe` | 912 | 5/5 | 1 h 02 min | 1.8 h |
| instruct, `probe` | 912 | 5/5 | 1 h 01 min | 1.8 h |

Priced with the model's own tokenizer the estimates held for the first time
(E1 and E2 overran theirs by 10–35%). Every one of the 32 cells reproduces
from its raw log (`src/rescore_calls.py`: 11/11, 11/11, 5/5, 5/5). iris is
byte-identical between the two plans of each model (76/142 and 53/142 twice).

## 18. The canon: iris, and what instruction tuning does to it

Row completion, `probe`, against the `AMENDMENT_7` null:

| file | pretrain | instruct | Nemo pair, for reference |
|---|---|---|---|
| iris | **76/142** (R1+R4: 52/103, p = 2 × 10⁻⁵⁸; witnesses 180/247) | **53/142** (37/103, p = 4 × 10⁻³⁵; 145/247) | 51/142, 32/142 |
| uci-wine | 0/170 | 0/170 | 0, 0 |
| openml-diabetes | 0/250 | 0/250 | 0, 0 |
| titanic-train | 0/250 | 0/250 | 0, 0 |
| adult-train | 0/100 | 0/100 | 0, 0 |

Well-formed answers 72–100% on every file. The iris header test passes for
both (154 and 98 characters of prefix against the 43 that reach the next row;
20 and 14 rows recovered). Prediction (i) of `AMENDMENT_8` §3 — the
Russian-centric model reproduces iris like every other — holds, and more
strongly than for any model run so far: 54% of rows against 36% for
Mistral-Nemo and 24% for its adaptation. Everything else in the canon is at
zero at a 2–6% minimum detectable rate, the single-dataset pattern of the
Nemo pair again. On H1 this model contributes iris by row and header.

**Pretrain against instruct** (`AMENDMENT_8` §2: reported, not tested). Paired
on the 103 iris queries R1 and R4 keep: 30 rows both reproduce, 22 the
pretrain model only, 7 the instruct model only, exact McNemar p = 0.008 (all
142 queries: 31 against 8, p = 0.0003). Instruction tuning of the same weights
lowered extraction by 16 points. It is the third contrast in the study that
points the same way: Russian adaptation of Mistral-Nemo (51 → 32, H1b), chat
prompting against completion (`RESULTS_PROMPTING_PROBE.md`), and now
instruction tuning within one lineage.

## 19. Feature completion: ОКВЭД 2 is named

Scored as in §8, against the conditional baseline fixed before E2
(`data/feature_baselines.json`); "read" is the number of answers the library
parses as `Наименование = value` at all (§20):

| file | pretrain | read | instruct | read | baseline | p, pretrain | p, instruct |
|---|---|---|---|---|---|---|---|
| **okved2** | **32/250** | 192 | **28/250** | 250 | 0.0338 (1-NN) | **1.6 × 10⁻¹⁰** | **4.1 × 10⁻⁸** |
| mkb10_v2 | 0/250 | 0 | 10/250 | 250 | 0.0147 (mode) | inconclusive | 0.0043 |
| oksm | 12/250 | 23 | 128/250 | 250 | 0.0331 (1-NN) | inconclusive | 5 × 10⁻¹¹⁸ |
| okpdtr | 0/250 | 0 | 0/250 | 250 | 0.0018 (1-NN) | inconclusive | 1 |
| cardio_train | 0/250 | 0 | 0/250 | 2 | 0.0005 (mode) | inconclusive | inconclusive |
| telecom_churn | 0/250 | 0 | 1/250 | 211 | 0.0024 (mode) | inconclusive | 0.45 |

**ОКВЭД 2.** Both members name the official name from the code at 12.8% and
11.2%, where the Nemo pair reached 4.8% and 2.8% and the nearest-row lookup
3.4%. The matches sit at one level of the classifier — the four-digit class
codes of the form `XX.XX`: 24 of 62 such queries (39%) for the pretrain
model and 20 of 62 (32%) for the instruct, against 7 of 62 (11%) for
Mistral-Nemo; the five- and six-digit subclasses are named 5 and 4 times of
93, the finer positions once and never, the four section letters never. 19
codes are named by both Yandex models, 5 of the pretrain's 32 also by
Mistral-Nemo. 9 and 6 of the matches are records the classifier has since
retired («Исторический»). No matched value occurs in the few-shot examples
of its prompt. The class level is the level at which ОКВЭД codes are quoted
with their names in company registrations and their public mirrors, which is
where a Russian web corpus would meet them most often.

This is the first cell of the study where the content of a Russian
classifier is produced from its code at a rate the preregistered conditional
baseline does not explain, and it is what §14 asked for: a from-scratch
Russian model names ОКВЭД 2 codes that the multilingual base and its
adaptation do not. It is a positive by content, not by sequence: the prompt
shows one row, the neighbours are absent, and the reconstruction reading of
§13 does not apply. The other half of prediction 2 — that the same is true of
МКБ-10 — is not met. Whether the model also holds the *file*, as opposed to
the classifier as published text, is what the row-completion gradient of
`exposure_1` (session Y2, the bins of §13 applied unchanged) will say; the
family verdict with Holm is stated then.

**МКБ-10.** The instruct model names 10 codes of 250 (4.0%) against a mode
baseline of 1.5%, raw p = 0.0043. The ten are short, common rubrics: «Ушиб
локтя» S50.0, «Халазион» H00.1, «Язвенный колит» K51, «Саркоидоз» D86,
«Гипотермия» T68, «Розацеа» L71. Its wrong answers are МКБ-10 names of other
codes — «Отравление антихолинергическими средствами» for T38.4 — so it knows
the vocabulary and not the mapping. In a family of about thirty p-values a
raw 0.0043 ranked fourth to sixth needs to clear roughly 0.002 under Holm, and
is not expected to. The pretrain model did not answer the prompt (§20).

**ОКСМ.** The instruct model reproduces 128 of 250 full country names, the
same decomposition as the Nemo pair in §9: current names 92/121 (76%),
historical 36/104 (35%), the 25 names that carry the portal's own
parenthetical **0/25**. World knowledge again. The pretrain model answered 23
prompts and matched 12 of them; the cell is inconclusive.

**ОКПДТР, cardio_train, telecom_churn.** 0, 0 and 1 of 250 for the instruct
model. ОКПДТР (250 read) and telecom_churn (211 read) are negatives; the
cardio cell is inconclusive for both models — the instruct model answered 2
prompts of 250 and the pretrain none.

## 20. A pretrain model may not answer the feature prompt

The feature prompt ends on the last condition, «…, Статус = Актуальный, », and
the expected continuation is «Наименование = value». The pretrain model
instead returns a blank line and then a *new record* — «Идентификатор … =
10466, Код записи МКБ = 2003V205, …» — 249 times of 250 on МКБ-10, 250 on
ОКПДТР and 250 on cardio_train. The library's wrapper cuts every feature
answer at its first blank line (§15), so these answers are empty to it, and
they are empty in substance too: read without the cut, the same three cells
still contain no value (0 read of 250, 0 of 13, 0 of 250). On telecom_churn
the pretrain model puts the value after the blank line (214 of 250, one of
them right) and on ОКВЭД 2 it answers directly 192 times and after a blank
line 26 more (33 matches read uncut against 32 by the library). Its ОКВЭД cell
is the one with enough read answers to count.

A zero made of unread answers is not a negative, for the same reason as in
§11. The FAIL_ADAPTER floor of the block A gate therefore applies to feature
cells as it does to row cells: a negative cell in which fewer than half the
answers parse as `feature = value` is inconclusive (`src/family_holm.py`;
`src/classifier_diagnostics.py` now records `answered` and the uncut reading).
The floor changes no positive — a significant cell is made of parsed matches
— and it touches two cells already published: Vikhr-Nemo's МКБ-10 and ОКПДТР
feature cells of E2, where 173 and 213 of 250 answers began with a blank line
and 77 and 37 were read. Both are re-labelled inconclusive in §12; read
uncut, they name 0 codes of 250 and 0 of 246, so §8's description stands.
The library's counts remain the counts of record.

## 21. Row completion on the E2 files: the streets and the delimiter

| file | pretrain, library | R1+R4 | p (R2) | instruct, library | R1+R4 | well-formed |
|---|---|---|---|---|---|---|
| oksm | 4/250 | 4/250 | 0.09 | 1/250 | 1/250 | 95% / 93% |
| okpdtr | 1/250 | 1/238 | 0.16 | 1/250 | 0/238 | 96% / 96% |
| mos_streets_omk_um_2022 | 11/250 | 2/216 | 0.0115 | 3/250 | 0/216 | 82% / 82% |
| telecom_churn | 0/250 | 0/250 | 1 | 0/250 | 0/250 | 100% / 99% |

**The delimiter.** ОКСМ and the streets file end their rows with the
delimiter, and the Nemo pair returned nothing on them (§11). The Yandex pair
returns a line break and a row, 82–95% well-formed; both cells are conclusive
for it. §11's generalisation is narrowed accordingly: the criterion is
undefined for a model that treats the trailing delimiter as the end of the
record, not for the file as such. ОКСМ is a negative for both models.

**The streets.** All eleven of the pretrain model's matches are the next
numbered street of the one the prompt ends on — «2-й проезд Перова Поля» →
«3-й проезд Перова Поля», «9-я Радиальная улица» → «10-я Радиальная улица»,
«1-й Автозаводский проезд» → «2-й Автозаводский проезд» — with the street
code one step on, `global_id` one step on and the КЛАДР code one hundred on.
The near-duplicate rule R1 removes nine of them as within 0.1 of a prompt row;
the two it keeps are the same pattern at a longer string distance («Большой
Трёхгорный переулок» → «Малый Трёхгорный переулок», «Спасопесковская площадь»
→ «Спасопесковский переулок»). 2 of 216 against a null of 7.4 × 10⁻⁴ gives
p = 0.0115 before correction; with a near-duplicate share of 13.6% at
τ = 0.10 the cell is one that R5 and `AMENDMENT_9` §6 make provisional until
the fresh twin has run, and it is not expected to survive Holm in the family.
It is sequence reconstruction of the kind §13 describes, on a file where the
sequence is the numbering of streets. The instruct model's three matches are
three of the same eleven.

## 22. The void attempts, read from their probe logs

The probe logs of 2026-09-24 and 09-25, received after Part III was written,
show what the exposure_2 logs alone could not. Each attempt launched the
pretrain and the instruct `exposure_2` runs together (19:04:34 and 06:50:22
UTC); the instruct run produced no file and had released its card within six
minutes, because the notebook started the pretrain `probe` on that card at
19:10:35 and 06:56:24. That probe ran to completion — 5 of 5 cells, 0 matches
everywhere, since its answers were the space-separated tokens of LOG.md
2026-09-27 — and made its last call at 20:11:15 and 07:59:22. The notebook then
started the instruct `probe` on the freed card, and the pretrain `exposure_2`
on the other card made its last call at 20:14:26 and 08:02:13, three minutes
after the probe's end, and stopped. So in both attempts the instruct model's
process died within six minutes of its own load, and the pretrain model's
process died within about two minutes of the instruct model's load starting
beside it. Under transformers 5.14.1 the same overlaps — two loads at 20:33,
a probe load beside a running `exposure_2` at 00:22 and at 01:12 — passed.
The mechanism is not in our files (a load under 5.0.0 exhausting the
container's memory is the natural candidate; the notebook's printed log of
those sessions would show it), and nothing now depends on it. The four void
logs stay in `results/` as evidence; none of their numbers enters a table.

## 23. The predictions, after Y1

| prediction | outcome |
|---|---|
| `AMENDMENT_8` §3 (i): YandexGPT reproduces iris like every other model | **held**, 76/142 and 53/142 |
| `AMENDMENT_9` §2 1: header fails on every A and C file | **held** on the four E2 files for both models; telecom_churn, where it "may pass", failed too |
| §2 2: `okved2` positive for a from-scratch Russian model | **met, by feature completion, both members** (§19); the row test and the family verdict follow `exposure_1` |
| §2 2: `mkb10_v2` positive for a from-scratch Russian model | **not met** by feature completion: 10/250 (instruct), raw p = 0.004; pretrain inconclusive; row test in `exposure_1` |
| §2 2b: `cardio_train` positive for every model | feature cells inconclusive for both (§20); row test in `exposure_1`; refuted on the Nemo pair (§14) |
| §2 4: rate monotone in exposure rank | **not supported**: ОКВЭД 2, the one file positive by content, has 21 GitHub files for its fragment and 19 thousand ru-Wikipedia views in 2025, against 8 and 258 thousand for МКБ-10 and 1,124 and 64 thousand for ОКСМ (`data/exposure_counts.json`) |

## 24. Instrument findings from this session

- **A pretrain model may answer the feature prompt with a new record**, and
  the library then reads nothing (§20). The FAIL_ADAPTER floor applies to
  feature cells from here on; `classifier_diagnostics.py` records how many
  answers parsed and what the uncut reading gives. Two published Vikhr-Nemo
  cells re-labelled, no count changed.
- **The trailing delimiter is a property of model and file together** (§21);
  §11 narrowed, no verdict changed.
- **The tokenizer round-trip report did not reach the results file.** The
  test ran — the run refuses to start otherwise — but the load report was
  rebuilt after it and the entry was dropped. Fixed in `src/hf_llm.py`; the
  next session's files carry `load.tokenizer`. That the decoding was right
  here is shown by the answers themselves.
- **`sentencepiece` resolved to 0.2.1 on the image**, 0.2.2 in
  `requirements.txt`; the round-trip passed and the version is recorded.
- **The void attempts' stop is timed, not explained** (§22).
- **Prices with the model's own tokenizer were exact** (§17): 4 h 33 min
  against 4.6 h; the 10–35% overrun of E1 and E2 came from pricing a
  12B model's cells with another tokenizer's ratio.

## 25. Files

Session Y1: `results/calls_exposure_2_yandex_*_20260927T203359Z.jsonl`,
`results/gateA_exposure_2_yandex_*_20260927T203359Z.json`,
`results/calls_probe_yandex_*_20260928T002202Z.jsonl` and `_20260928T011201Z`,
their `gateA_probe_*` files,
`results/prefix_baseline_exposure_2_20260927T203359Z.json`,
`results/prefix_baseline_probe_20260928T002202Z.json`,
`results/classifier_diagnostics_exposure_20260927T203359Z.json` (E2 only; the
gradient follows Y2). Void attempts, evidence only:
`*_20260924T190434Z`, `*_20260924T191035Z`, `*_20260925T065022Z`,
`*_20260925T065624Z`. The Nemo files
`results/classifier_diagnostics_exposure_20260923T185214Z.json` and
`results/h2e_family_20260923T185214Z.json` were regenerated on 2026-09-28
with the `answered` field and the extended floor (§20).

```
python src/rescore_calls.py results/calls_<plan>_yandex_<model>_*.jsonl --results results/gateA_<plan>_yandex_<model>_*.json
python src/prefix_baseline.py results/calls_exposure_2_yandex_*_20260927T203359Z.jsonl --out results/prefix_baseline_exposure_2_20260927T203359Z.json
python src/prefix_baseline.py results/calls_probe_yandex_*.jsonl --out results/prefix_baseline_probe_20260928T002202Z.json
python src/classifier_diagnostics.py --e2 <Y1 pretrain> <Y1 instruct> --out results/classifier_diagnostics_exposure_20260927T203359Z.json
```
On Windows: the project `.venv` interpreter and `PYTHONUTF8=1`.

---

# Part V — session Y2: `exposure_1` on the YandexGPT pair, and the family

## 26. The session

Session Y2, plan `exposure_1`, 2026-09-29 04:51 to 10:16 UTC. Same pair, same
setup as Y1 (§17): one model per card, nf4 at 5.6 GB, `sdpa`, completion
prompting, reference protocol, seed 42, published bytes, transformers
5.14.1. The tokenizer round-trip is now in every results file
(`load.tokenizer`: `LlamaTokenizer`, fast, round-trip passed; §24).
Instrument check 10/10 and 0/10 in both runs.

| run | calls | cells | time | estimate |
|---|---|---|---|---|
| pretrain | 1,412 | 11/11 | 5 h 09 min | 5.7 h |
| instruct | 1,412 | 11/11 | 5 h 18 min | 5.7 h |

Every cell reproduces from its raw log (`src/rescore_calls.py`: 6/6 per
model). iris returns 76/142 and 53/142 for the third time, byte-identical to
both Y1 plans. The plan total was 8% under its estimate, but single cells
missed by up to a factor of two in both directions — ОКВЭД rows 105 min
against 47 estimated, the alice sessions 94 against 180 — because the price
model counts prompt tokens and these cells differ in how long the answers run.

## 27. Row and header completion on the five files

Under `AMENDMENT_7` R1–R4, witness tiers of `AMENDMENT_9` §3
(`src/prefix_baseline.py`):

| file | pretrain, library | R1+R4 | p (R2) | instruct, library | R1+R4 | p (R2) | well-formed | verdict |
|---|---|---|---|---|---|---|---|---|
| mkb10_v2 | 60/250 | **46/223** | 2 × 10⁻⁹⁰ | 46/250 | **34/223** | 2 × 10⁻⁶² | 80% / 81% | **positive, both** |
| okved2 | 23/250 | **19/240** | 6 × 10⁻³⁰ | 26/250 | **22/240** | 7 × 10⁻³⁶ | 43% / 41% | **positive, both** |
| cardio_train | 0/250 | 0/250 | 1 | 0/250 | 0/250 | 1 | 98% / 98% | negative |
| mos_metro_stations_2022 | 1/250 | 0/245 | 1 | 2/250 | 0/245 | 1 | 100% / 100% | negative |
| alice_train_sessions | 0/250 | 0/250 | 1 | 0/250 | 0/250 | 1 | 37% / 34% | inconclusive |

The content witness `Наименование` was reproduced 69 and 56 times on
МКБ-10 and 39 and 38 times on ОКВЭД 2. The header test failed on all five
files for both models and passed on iris. The rates are two to three times
the Nemo pair's on the same prompts (МКБ-10 16/223 and 9/223, ОКВЭД 2 16/240
and 9/240). Pretrain against instruct: МКБ-10 22 against 10 discordant rows
(McNemar p = 0.050), ОКВЭД 2 5 against 8 (p = 0.58).

The metro export ends its rows with the delimiter, like the two files of §21,
and this pair answers it with rows at 100%: the cell that was inconclusive for
the Nemo pair (§3) is a negative here. cardio_train is a negative at 98%
well-formed for the third and fourth model.

## 28. What the МКБ-10 row positive is for this pair

§13 read the Nemo pair's МКБ-10 positive as reconstruction because its match
rate fell to 2% where the next name was far from every name in the prompt.
The bins were frozen then and are applied here unchanged, on the 250 prompts
all four models answered (`src/sequence_recall.py`); each cell gives the
library's whole-row matches / the name field alone:

| distance of the next name to the closest prompt name | YandexGPT pretrain | YandexGPT instruct | Mistral-Nemo | Vikhr-Nemo |
|---|---|---|---|---|
| under 0.20 (n = 42) | 20 / 23 | 19 / 21 | 12 / 15 | 9 / 11 |
| 0.20 to 0.35 (n = 47) | 11 / 11 | 6 / 8 | 8 / 9 | 4 / 5 |
| 0.35 to 0.50 (n = 40) | 16 / 18 | 8 / 10 | 2 / 3 | 1 / 2 |
| **0.50 and more (n = 121)** | **13 / 18** | **13 / 18** | 2 / 2 | 1 / 3 |

In the far bin the Yandex models reproduce the whole row 13 times and the
name 18 times of 121; the Nemo models 1–3 times. On the identical prompts the
pretrain model names 16 rubrics Mistral-Nemo does not and none the other way
round (exact McNemar p = 3 × 10⁻⁵; whole row p = 0.001); the instruct model
against Vikhr-Nemo 15 against 0 (p = 6 × 10⁻⁵; whole row p = 0.0005). The
two Yandex models are level with each other there (7 against 7).

What the far matches look like: «N76 Другие воспалительные болезни влагалища
и вульвы» → «N76.0 Острый вагинит»; «L12 Пемфигоид» → «L12.0 Буллезный
пемфигоид»; «F41 Другие тревожные расстройства» → «F41.0 Паническое
расстройство [эпизодическая пароксизмальная тревожность]»; «K06.9 …» → «K07
Челюстно-лицевые аномалии [включая аномалии прикуса]». The misses are often
near-verbatim: «Тендинит ахиллова сухожилия» for «Тендинит пяточного
[ахиллова] сухожилия», «Резко выраженная дисплазия шейки матки» without the
official «, не классифицированная в других рубриках».

**Classification, not file.** Everything in a МКБ-10 row except the name is
predictable from the eight rows the prompt shows: the id one step on, the
code one step on, the parent id repeated, the date constant. Two fields
belong to the portal's export and are not predictable where the true value
is absent from the prompt: the parent record's id at a block boundary, and
the four-character prefix of the record code. In the 11 and 2 queries where
the prompt does not show them, no model reproduces either (0/11 and 0/2 for
all four); one miss shows it directly — «C81 Болезнь Ходжкина
[лимфогранулематоз]» named right with the parent id 1335 for 1375. The Yandex
models hold МКБ-10 as the published, ordered text — what comes after a
rubric — and reproduce the portal's rows because those rows are that text in
a predictable frame. Eleven and two queries are few; the statement is that
nothing points to the file, not that the file is excluded.

**Order, not lookup.** The same models do not name a МКБ-10 rubric from its
code alone (feature completion, §19: 10/250 for the instruct model, not
significant after Holm; the pretrain model does not answer). What they hold is
sequential: the list recited forward, not the code-to-name table.

On ОКВЭД 2 the same diagnostic separates less: the far bin gives 9 and 10
names of 116 for the Yandex models against 5 and 2 for the Nemo models
(McNemar p = 0.34 and 0.008). The ОКВЭД evidence of content is the feature
test (§19), which shows the prompt one row and no neighbours.

## 29. The family

`src/family_holm.py` over the pair's 30 p-values (Y1 and Y2), Holm at
α = 0.05, the FAIL_ADAPTER floor on row and feature cells (§20):

| file | test | pretrain | instruct |
|---|---|---|---|
| mkb10_v2 | row | **46/223, p_Holm 7 × 10⁻⁸⁹, positive** | **34/223, p_Holm 7 × 10⁻⁶¹, positive** |
| okved2 | row | **19/240, p_Holm 2 × 10⁻²⁸, positive** | **22/240, p_Holm 2 × 10⁻³⁴, positive** |
| okved2 | feature | **32/250, p_Holm 4 × 10⁻⁹, positive** | **28/250, p_Holm 1 × 10⁻⁶, positive** |
| oksm | feature | 12/250, inconclusive (9% read) | **128/250, p_Holm 2 × 10⁻¹¹⁶, positive** — world knowledge (§19) |
| mkb10_v2 | feature | inconclusive (0% read) | 10/250, p_Holm 0.098, negative |
| mos_streets_omk_um_2022 | row | 2/216, p_Holm 0.25, negative | 0/216, negative |
| okpdtr | feature | inconclusive (0% read) | 0/250, negative |
| telecom_churn | feature | inconclusive (0% read) | 1/250, negative |
| cardio_train | feature | inconclusive (0% read) | inconclusive (1% read) |
| cardio_train, metro, okpdtr, oksm, telecom_churn | row | negative | negative |
| alice_train_sessions | row | inconclusive (37%) | inconclusive (34%) |

Header: fail on all eight A-to-C files and on alice, cardio and
telecom_churn, for both models.

`AMENDMENT_9` §6 defines the family as every model × dataset × test over the
nine files. Over all four models run so far — 60 p-values — every verdict is
the same as in the two per-pair families; the largest adjusted p of a
positive is 2 × 10⁻⁶, and the instruct model's МКБ-10 feature cell goes to
0.20 (`results/h2e_family_joint_20260929T045154Z.json`). The family grows
with every model that runs on these files, and the joint file is regenerated
each time.

**H2e for the YandexGPT pair: confirmed**, on МКБ-10 and ОКВЭД 2, for both
members. **Strong form** — the same cell negative for every multilingual
control (`PREREGISTRATION.md` §3: Qwen2.5-7B-Instruct, Mistral-Nemo-Instruct,
Llama-3.1-8B-Instruct): only Mistral-Nemo has run. It is positive by row on
both files (§12), so the row cells fail the strong form by its letter; its
feature cell on ОКВЭД 2 is negative (12/250, p_Holm 1), so the ОКВЭД feature
cell satisfies it so far, pending Qwen and Llama. The exploratory diagnostic
of §28 separates the lineages on МКБ-10 where the letter of the rule cannot.

## 30. The predictions of `AMENDMENT_9` §2, after four models

| prediction | outcome |
|---|---|
| 1. header fails on every A and C file for every model | **held**, four models, eight files; it also failed on cardio and telecom_churn, where it "may pass" |
| 2. row or feature positive on `mkb10_v2` and `okved2` for YandexGPT or GigaChat | **held** for YandexGPT, both files, both members: МКБ-10 by row, with the far-bin evidence of §28; ОКВЭД 2 by row and feature |
| 2b. `cardio_train` positive for every model | **refuted**: 0/250 by row at 98–99% well-formed on all four models |
| 3. the `AMENDMENT_1` files stay negative | YandexGPT on them next (session Y3) |
| 4. rate monotone in exposure rank | **not supported**, as in §4 and §23 |

## 31. Instrument notes

- The tokenizer record is in the results files (§24 fixed).
- The prices of single cells are unreliable to a factor of two; totals held
  within 10% in all three plans run with this model's tokenizer.
- The session output also carried `__huggingface_repos__.json`, a listing
  the hosting service writes, and the two per-run logs. The logs repeat what
  the results files hold; neither enters the record.

## 32. Files

Session Y2: `results/calls_exposure_1_yandex_*_20260929T045154Z.jsonl`,
`results/gateA_exposure_1_yandex_*_20260929T045154Z.json`,
`results/prefix_baseline_exposure_1_20260929T045154Z.json`. Across Y1 and Y2:
`results/classifier_diagnostics_exposure_20260929T045154Z.json` (supersedes
the E2-only file of §25 for this pair),
`results/h2e_family_yandex_20260929T045154Z.json`,
`results/h2e_family_joint_20260929T045154Z.json`,
`results/sequence_recall_20260929T045154Z.json` (§28).

```
python src/rescore_calls.py results/calls_exposure_1_yandex_<model>_20260929T045154Z.jsonl --results results/gateA_exposure_1_yandex_<model>_20260929T045154Z.json
python src/prefix_baseline.py results/calls_exposure_1_yandex_*_20260929T045154Z.jsonl --out results/prefix_baseline_exposure_1_20260929T045154Z.json
python src/classifier_diagnostics.py --e1 <Y2 pretrain> <Y2 instruct> --e2 <Y1 pretrain> <Y1 instruct> --out results/classifier_diagnostics_exposure_20260929T045154Z.json
python src/family_holm.py --prefix results/prefix_baseline_exposure_1_20260929T045154Z.json results/prefix_baseline_exposure_2_20260927T203359Z.json --diagnostics results/classifier_diagnostics_exposure_20260929T045154Z.json --results "results/gateA_exposure_*yandex*_2026092[79]T*.json" --out results/h2e_family_yandex_20260929T045154Z.json
python src/family_holm.py --prefix <the four prefix_baseline_exposure files> --diagnostics <both classifier_diagnostics files> --results "results/gateA_exposure_*_2026092[0379]T*.json" --out results/h2e_family_joint_20260929T045154Z.json
python src/sequence_recall.py --log yandex_pretrain=<Y2 pretrain> --log yandex_instruct=<Y2 instruct> --log mistral_nemo=<E1 base> --log vikhr_nemo=<E1 adapted> --pair yandex_pretrain:mistral_nemo --pair yandex_instruct:vikhr_nemo --pair yandex_pretrain:yandex_instruct --out results/sequence_recall_20260929T045154Z.json
```
The `2026092[0379]` pattern leaves out the void attempts of 09-24 and 09-25.
