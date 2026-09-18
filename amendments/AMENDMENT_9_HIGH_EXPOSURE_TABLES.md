# Amendment 9 to PREREGISTRATION.md — high-exposure Russian tables

**Date:** 2026-09-18. **Scope:** the exposure hypothesis that `AMENDMENT_8` §1
stated in advance — that the H2 null of session C1 reflects how rarely the
tested files occur in public text, not what Russian adaptation does to a model
— is given the files it needs. Nine tables that the Russian internet repeats
are frozen here with their hashes, their witness columns and a measured
exposure covariate, **before any model has seen any of them**. They form a
second, pre-declared family of the H2 surface (H2e, §2) with its own
correction; the H2 verdict on the `AMENDMENT_1` files is reported unchanged.
No test, prompt, threshold, α or model changes.

**Transparency.** The files were chosen after session C1 was negative on every
Russian file. The criterion for choosing them — public exposure — was written
into `AMENDMENT_8` before any candidate was collected, and the predictions in
§2 are written before any run. Every candidate that was considered is listed
in `LOG.md` (2026-09-18) with the model-free number that admitted or rejected
it (`src/profile_candidates.py`); the rejections are structural (multi-line
records, fewer than 300 rows, a post-cutoff data version, personal data,
XML), not outcome-based, because there is no outcome yet.

---

## 1. Why these files

Bordt et al. explain iris by its ubiquity in notebooks and textbooks; Kandpal
et al. (arXiv:2202.06539) measure the mechanism — a sequence seen once in
training is almost never regurgitated, one seen ten times is regurgitated
orders of magnitude more often. The five Russian files of `AMENDMENT_1` are
single-source portal exports: GitHub holds 0–31 files that contain one of
their rows (§4), and every cell was negative. A "Russian iris" is not a file
on a portal; it is a table the Russian web reprints thousands of times:
federal classifiers, reference lists with an encyclopaedia counterpart, and
the datasets of the Russian machine-learning courses whose notebooks print
their first rows. Nine such tables, in three groups:

| group | files | what the internet repeats | enters |
|---|---|---|---|
| A, classifiers | `mkb10_v2`, `okved2`, `oksm`, `okpdtr` | the code → official name content, reprinted by every medical, business, HR site and by GitHub dumps | H2e; satisfies §4 |
| B, course canon | `cardio_train`, `alice_train_sessions`, `telecom_churn` | the CSV file itself, through Kaggle and the mlcourse.ai repository (10.7k stars) — Bordt's mechanism | H2e only: English headers fail §4's language clause |
| C, reference lists | `mos_metro_stations_2022`, `mos_streets_omk_um_2022` | the content (stations → lines, streets → codes) with a Wikipedia list of 161k and 95k views a year | H2e; satisfies §4 |

Group A carries a caveat that decides how a result may be read. What the
internet repeats is the code and the name; the portal's CSV shape — the
`Идентификатор`, `Дата актуализации`, `Статус` columns, the BOM, the column
order — was published by one portal. A positive on such a file is therefore
memorization of a reference table's content, not of the file; a negative
refutes "the model knows this portal's dump of ОКВЭД", not "the model knows
ОКВЭД". The header test is predicted to fail on every A and C file for the
same reason (§2). Group B has no such caveat: its files circulate as files.

---

## 2. Hypothesis H2e, with its predictions

**H2e.** A Russian-centric model reproduces verbatim rows, or content-witness
values, of a high-exposure Russian table at a rate above the `AMENDMENT_7`
null, and across tables the rate rises with exposure.

*Confirmed* if at least one (Russian-centric model × A-or-C file) cell is
positive under §6. *Strong form:* the same cell is negative for every
multilingual or Western control on the same file. *Refuted* if every A and
C cell is negative for every Russian-centric model **while** the group-B
anchor `cardio_train` is positive for at least one model (the instrument
demonstrably reaches a course-canon file at these sizes) and the iris anchor
fires in the same session — in which case the exposure hypothesis for the H2
null is not supported at the model sizes tested, and the null is reported as
a property of Russian tables at 8–20B parameters, not of the files chosen.
If cardio and iris are also negative for a model, that model makes no claim
(`PREREGISTRATION.md` §8).

Predictions, written before any run:

1. The header test fails on every A and C file for every model (the header
   line exists in one portal's export only); it may pass on `cardio_train`
   and `telecom_churn`, whose first rows are printed in hundreds of
   notebooks.
2. Row completion or feature completion is positive on `mkb10_v2` and
   `okved2` for at least one of YandexGPT-5-Lite-8B and GigaChat-20B-A3B, the
   two models trained from scratch on a Russian-heavy corpus, and positive on
   `cardio_train` for every model including the Western bases.
3. The `AMENDMENT_1` files remain negative for every model, as in C1.
4. Across the twenty-one registered datasets, the row-completion rate under
   R1–R4 is monotone in the exposure rank of §4 (Spearman ρ > 0, exploratory,
   reported with its p-value and no threshold).

Relation to H2: the A and C files meet §4's criteria (Russian headers or
values, ≥ 300 rows, a high-entropy feature, dated pre-2023 evidence), so they
are Russian pre-cutoff datasets in the preregistration's sense. They are not
merged into the `AMENDMENT_1` family because they were chosen after C1: they
form family H2e with its own Holm correction, and the paper reports the two
families side by side. Group B enters H2e only; its files were never
"Russian-language" datasets and are not counted toward H2.

---

## 3. Witness columns in two tiers (extends `AMENDMENT_7` §3)

`AMENDMENT_7` R3 admits as a witness only a value that no public source
outside the file supplies. For a file that *is* a public classifier, that
rule leaves nothing, and the tension is the point of this amendment. Two
tiers are therefore designated, both before any run, in
`data/witness_columns.json` (`_tiers`):

- **file tier** — a value that exists in this file only: a portal-assigned
  identifier. Reproducing it means the file itself was seen.
- **content tier** — the official name, or the measurement, that every copy
  of the table carries. Reproducing it means the table's content was
  memorized, whichever copy the model saw.

| dataset | file tier | content tier (also the feature-completion feature) | not witnesses, and why |
|---|---|---|---|
| mkb10_v2 | — | `Наименование` | `Код записи МКБ` (positional surrogate), `Код состояния по МКБ` (public code), dates, status, ids |
| okved2 | — | `Наименование` | `Код`, `Код позиции` (public codes), `Раздел`, dates, status, `Идентификатор` (sequential) |
| oksm | — | `Полное наименование по ОКСМ`, `Краткое наименование по ОКСМ` | `Цифровой код`, `Альфа-2`, `Альфа-3` (ISO 3166), dates, status |
| okpdtr | — | `Наименование` | `Код` (public), `Код ОКЗ` (published mapping), `Код ЕТКС/Категория`, dates, status |
| mos_metro_stations_2022 | `global_id` | `Наименование линии`, `Район` | `Код` (portal code), status, the empty and flag columns |
| mos_streets_omk_um_2022 | `global_id` | `UM_CODE` | `UM_KLADR` (КЛАДР, a public classifier), `UM_TRANS` (a transliteration convention), type codes |
| cardio_train | — | `age` (days), `weight` | `id` (sequential), the binary and ordinal columns |
| alice_train_sessions | — | `time1`, `time2`, `time3` | `session_id` (sequential), `siteN` (dictionary ids), `target` |
| telecom_churn | — | `Total day minutes`, `Total eve minutes`, `Total night minutes`, `Total intl minutes` | the `charge` columns (a fixed multiple of the minutes), `State`, `Area code`, the plan flags, `Churn` |

Rule: a row-completion cell is positive only if at least one witness of
either tier, absent from the prompt, is reproduced as a complete field of
the scored line (R3 unchanged); the tier of every reproduced witness is
reported. A positive carried by the content tier alone is reported as
"reference table memorized"; one carried by the file tier as "file seen".
For `cardio_train` the content witnesses have low cardinality on their own
(weight: 290 values in 70,000 rows), so the two-tier report also gives, for
that file, the count of rows whose *pair* (`age`, `weight`) is reproduced —
a descriptive supplement, not part of the rule.

---

## 4. The exposure covariate (expedition E1 of `AMENDMENT_8`)

"Exposure" is how often a table occurs in public text a model could have
been trained on. It is measured by one rule for every registered dataset,
canon and Russian alike, by `src/exposure_counts.py`, and stored in
`data/exposure_counts.json`; the values below were measured on 2026-09-18,
before any run on the new files.

- **GitHub, file name** — the number of files GitHub code search returns for
  the table's canonical file name, where it has one (`iris.csv`,
  `cardio_train.csv`, …). Authenticated search, forks and archived
  repositories excluded, files over 384 KB not indexed: a floor.
- **GitHub, two fragments** — the same count for two strings taken from the
  file by rule: the records at positions 10 and ⌊N/2⌋ (header excluded; for
  the Moscow exports the Russian header line is skipped). A table that
  circulates as a CSV file contributes its whole physical line
  (`full_line`); a classifier, whose content is reprinted without its CSV
  shape, contributes the content-witness field alone (`name_column`). The
  two modes are not comparable with each other and are reported as two
  columns, never pooled.
- **Wikipedia** — the 2025 calendar-year user pageviews of the Russian
  Wikipedia article whose subject the table is (canonical title, not a
  redirect), where such an article exists.

| dataset | group | mode | GitHub: file name | GitHub: fragment 1 | GitHub: fragment 2 | ru-Wikipedia 2025 |
|---|---|---|---|---|---|---|
| iris | canon | full_line | 131,840 | 26,048 | 26,688 | 6,771 |
| adult-train | canon | full_line | 12,256 | 78 | 15 | — |
| california-housing | canon | full_line | 45,184 | 37 | 5 | — |
| openml-diabetes | canon | full_line | 53,504 | 12,448 | 16,352 | — |
| uci-wine | canon | full_line | 16,416 | 3,184 | 3,168 | — |
| titanic-train | canon | full_line | 76,544 | 30,464 | 32,448 | 596,503 |
| hflabs_city | ru_pre_cutoff | full_line | 36,608 | 26 | 31 | 846,543 |
| govdomains | ru_pre_cutoff | full_line | 3 | 0 | 0 | — |
| mos_zemelnye_uchastki | ru_pre_cutoff | full_line | — | 0 | 0 | — |
| mos_torgovye_obekty | ru_pre_cutoff | full_line | — | 0 | 0 | — |
| russian_retail | ru_pre_cutoff | full_line | — | 0 | 0 | — |
| trudvsem_vacancies_2026 | fresh_control | full_line | — | 0 | 0 | — |
| mkb10_v2 | ru_exposure | name_column | — | 8 | 1 | 257,778 |
| okved2 | ru_exposure | name_column | — | 21 | 73 | 19,205 |
| oksm | ru_exposure | name_column | — | 1,124 | 4,256 | 63,535 |
| okpdtr | ru_exposure | name_column | — | 4 | 2 | 4,924 |
| mos_metro_stations_2022 | ru_exposure | name_column | — | 417 | 918 | 161,095 |
| mos_streets_omk_um_2022 | ru_exposure | name_column | — | 32 | 15 | 95,406 |
| cardio_train | ru_exposure | full_line | 1,824 / 207 | 8 | 0 | — |
| alice_train_sessions | ru_exposure | full_line | 382 | 0 | 0 | — |
| telecom_churn | ru_exposure | full_line | 1,656 | 507 | 692 | — |

(`python src/make_amendment_9_tables.py --exposure`; `—` = no canonical file name, no article, or not measured)

Two readings the table forces. A `name_column` count is the frequency of the
name as a string, which for a country (`oksm`) or a metro station includes
every use outside the table; for those two files it overstates the table's
own exposure and is read together with the Wikipedia column. And the
`full_line` counts of the course files show where exposure sits: the tenth
row of `cardio_train` is in 8 files and its middle row in none, because
notebooks print the head of a table, not its middle — which is what the
header test measures and the row-completion test, sampling windows from the
whole file, does not.

The covariate is used two ways, both descriptive: as the x-axis of the
exposure figure (rate under R1–R4 against the fragment count, one point per
model × dataset), and as the rank in prediction 4 of §2. No threshold on it
enters any verdict. The earlier infini-gram counts over Dolma v1.7
(`AMENDMENT_8`) are kept as a third, English-only column.

---

## 5. Files frozen

Registered in `data/registry.json` (group `ru_exposure`) by
`src/prepare_exposure_datasets.py`, which copies the published bytes without
decoding them; the model-free profile is `data/ru_exposure_profile.json`
(`src/profile_candidates.py`) and the null and minimum detectable rate are
recomputed in `data/detectability.json`. Tests run on the `raw` variant only.

| dataset | source, version | published | sha256 | rows × cols | bytes | header + row 1 | null (R2) | MDR at 250 | near-dup τ=0.10 |
|---|---|---|---|---|---|---|---|---|---|
| mkb10_v2 | data.gov.spb.ru/irsi/7815000870-MKB_new/versions/1/export_data/ | 2019-05-23 | `1557d601328f592f…` | 14,772 × 11 | UTF-8 with BOM, `,` | 350 ✓ | 1.0e-03 | 1.2% | 9.4% |
| okved2 | data.gov.spb.ru/irsi/7815000870-OKVED_new/versions/19/export_data/ | 2020-06-22 | `191ed7dbd94ea284…` | 2,986 × 13 | UTF-8 with BOM, `,` | 929 ✗ | 1.0e-03 | 1.2% | 6.2% |
| oksm | data.gov.spb.ru/irsi/7815000870-OKSM/versions/4/export_data/ | 2019-07-29 | `0ec05a85c0da136c…` | 453 × 13 | UTF-8 with BOM, `,` | 345 ✓ | 6.7e-03 | 2.7% | 0.0% |
| okpdtr | data.gov.spb.ru/irsi/7815000870-OKPDTR/versions/1/export_data/ | 2019-05-18 | `1889a6c55d21dd67…` | 8,182 × 13 | UTF-8 with BOM, `,` | 298 ✓ | 7.3e-04 | 1.2% | 4.4% |
| mos_metro_stations_2022 | data.mos.ru/opendata/1488 | 2022-12-19 | `ade760f5a2b6fb5a…` | 306 × 12 | UTF-8, `;` | 401 ✓ | 1.0e-02 | 3.1% | 1.7% |
| mos_streets_omk_um_2022 | data.mos.ru/opendata/2044 | 2022-11-18 | `4f8fa632af0671c9…` | 5,406 × 10 | UTF-8, `;` | 226 ✓ | 7.4e-04 | 1.2% | 12.7% |
| cardio_train | www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset | 2019-01-20 | `21a705d23381b0df…` | 70,000 × 13 | UTF-8, `;` | 119 ✓ | 4.3e-05 | 0.6% | 0.0% |
| alice_train_sessions | www.kaggle.com/c/catch-me-if-you-can-intruder-detection-through-webpage-session-tracking2 | 2017-09-11 | `fdf265b85dfdd73c…` | 253,561 × 22 | UTF-8, `,` | 185 ✓ | 3.0e-05 | 0.6% | 0.0% |
| telecom_churn | raw.githubusercontent.com/Yorko/mlcourse.ai/0b6125c37cf6693ae0a698913456c3cf56058e54/data/telecom_churn.csv | 2021-12-09 | `5d0aecc74b6feda8…` | 3,333 × 20 | UTF-8, `,` | 415 ✓ | 9.0e-04 | 1.2% | 0.0% |

(`python src/make_amendment_9_tables.py --files`)

Provenance, per group:

- **St Petersburg (A).** `data.gov.spb.ru`, dataset pages
  `/irsi/<slug>/`; each file is the CSV inside the zip that
  `/irsi/<slug>/versions/<n>/export_data/` serves, at the latest version
  dated before 2023-01-01 in the portal's version list, and the CSV's own
  modification time inside the zip is that version's date. `src/fetch_data.py`
  re-fetches all four and verified the frozen hashes on 2026-09-18. The
  portal states no licence.
- **Moscow (C).** `data.mos.ru` keeps the full release history of a dataset
  and regenerates the export of any listed release on request
  (`odataExports/export` → `odataExports/status` → `MEDIA/getFile`,
  LOG.md 2026-09-18). The frozen files are the releases of 2022-12-19 (metro,
  version 3 release 14, 305 stations) and 2022-11-18 (streets, version 1
  release 62, 5,405 streets), regenerated on 2026-09-18: provenance is
  "release of that date, exported now", and the 2022 metro layout has no
  GUID column (the 2026 layout does). Each export carries two header lines,
  English then Russian; the library reads the Russian one as the first row.
  Licence reported as CC BY 4.0 from the portal footer, not verified from a
  machine-readable page. The portal refuses foreign connections, so the two
  files travel with the run as attached input, hash-checked like the fresh
  control.
- **Kaggle and mlcourse.ai (B).** `cardio_train.csv` is byte-identical to
  Kaggle's file (contentMD5 `ccRqcPe6nj5HNq5z7NW6yQ==` in the Wayback capture
  of 2019-07-15) and to `data/mlbootcamp5_train.csv` of `Yorko/mlcourse.ai`
  at commit `0b6125c3` (2021-12-09); licence "Unknown" on Kaggle, origin
  Mail.Ru ML Boot Camp V (2017), unattributed there. `train_sessions.csv` is
  the file of Kaggle in-class competition 7173 (launched 2017-09-11, Wayback
  2022-12-08); Kaggle serves competition data to accounts only, so it travels
  with the run. `telecom_churn.csv` is the raw file of `Yorko/mlcourse.ai` at
  the same pinned commit.

Row counts in the registry include the Russian header line of the two Moscow
exports (306 and 5,406 lines for 305 and 5,405 records).

---

## 6. Decision rules

Nothing in `PREREGISTRATION.md` §5 or `AMENDMENT_7` changes; three points
are made explicit for these files.

- **R4 applied to the Moscow exports:** a row-completion query whose target
  is the Russian header line is not a record and is excluded from count and
  denominator, exactly as fragment lines are on govdomains.
- **R5 applied to `mos_streets_omk_um_2022`:** its near-duplicate share at
  τ = 0.10 is 12.7% (alphabetical order), above the 10% that R5 names, so a
  positive row-completion verdict on it is reported as provisional until a
  construction-matched fresh control has run under the same plan. No other
  file of this amendment exceeds 10% at τ = 0.10; the τ-curve is reported for
  all.
- **Feature completion on a name column:** the §5 baseline is the best of
  mode / LR / GBT predicting the feature from the other columns. For
  `Наименование` the mode rate is 1/N and no classifier predicts a name from
  dates and status, so the effective null is the rule-of-three floor 3/W of
  R2; the baseline is nonetheless computed and reported.
- **Correction:** Holm within family H2e = every model × dataset × test
  p-value over the nine files, α = 0.05, separately from the `AMENDMENT_1`
  family. The header test contributes a verdict, not a p-value, as in §5.

---

## 7. Sessions, cost, validation

Two plans in `src/run_repro.py`, each priced by `src/price_plan.py` against
the cost model fitted to the C1 logs (the C1 estimate of 7.1 h ran in 7.4 h):

| plan | cells | conservative estimate per model |
|---|---|---|
| `exposure_1` | iris anchor; header + row on `mkb10_v2`, `cardio_train`, `alice_train_sessions`, `mos_metro_stations_2022`; row on `okved2` (its header + first row is 929 characters, over the 500-character window, `AMENDMENT_6` §3) | 6.0 h |
| `exposure_2` | iris anchor; header + row on `oksm`, `okpdtr`, `mos_streets_omk_um_2022`, `telecom_churn`; feature completion on the six files whose content witness is a name or a measurement | 4.9 h |

**Session rule, from this amendment on:** a hosted session is planned at no
more than 10 hours by the conservative estimate, leaving two of Kaggle's
twelve as reserve; a plan that prices above that is split.

Session E1 (`notebooks/session.json`) runs `exposure_1` on the Mistral-Nemo ↔
Vikhr-Nemo pair, one model per accelerator, seed 42, completion prompting,
reference protocol — the pipeline that C1 validated. The same plans then run
on the YandexGPT and GigaChat pairs and on the controls in the order of
`AMENDMENT_8` §5; the exposure figure needs every model on the same files.

Instrument check, run on this machine before the freeze was committed
(`results/validation/gateA_exposure_*`): the perfect memorizer scores 100% on every cell of both plans (11 cells of `exposure_1`, 16 of `exposure_2`: every header test passes, every row and feature count is n/n) and the format-echo mock scores zero on every cell. The perfect mock's own parser of feature-completion prompts was repaired first — it split conditioning fields on `, ` and lost the rows whose values contain a comma (54% of ОКВЭД rows), scoring 162/250 there — so that the ceiling it reports is the instrument's, not the mock's (`src/mock_llm.py`, LOG.md 2026-09-18).

---

## 8. What does not change

Hypotheses H1–H4 and their decision rules; the H2 family of `AMENDMENT_1`
and its verdict from C1; the four tests and their prompts; the primary
measure; the models of `AMENDMENT_8`; the reporting rules of `AMENDMENT_5`;
the near-duplicate rule of `AMENDMENT_7`, which this amendment applies and
does not alter.
