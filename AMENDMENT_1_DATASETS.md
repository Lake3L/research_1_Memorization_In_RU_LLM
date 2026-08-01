# Amendment 1 to PREREGISTRATION.md — dataset list frozen

**Date:** 2026-08-01. **Status:** frozen before any model was queried on any of these files.

PREREGISTRATION.md §4 committed to freezing the Russian dataset list, its canonicalisation and SHA-256 hashes in a dated amendment before any model saw them. This is that amendment. It is generated from `data/registry.json` by `src/make_amendment.py`, so the hashes below are the hashes of the files on disk, not transcribed by hand.

## Admission to H3

§4 admits a dataset on row count, language, an available high-entropy feature and evidence of publication date. Those criteria say nothing about whether the classification task the dataset defines is learnable at all — and a dataset whose task is not learnable cannot support H3, because few-shot accuracy on it would be uninformative whether or not the model memorized it. We therefore add one criterion, decided before any model was run: **a dataset enters H3 only if logistic regression or gradient boosting beats the majority class by at least 5 accuracy points** under stratified shuffled 5-fold cross-validation (`src/check_dataset.py`).

Datasets that fail this remain in H1/H2. The memorization tests ask whether the model reproduces rows verbatim; they need no target and are unaffected.

## Frozen datasets

### Russian, published before 2023-01-01

**`hflabs_city`** — 1117 rows × 24 columns

- source: https://raw.githubusercontent.com/hflabs/city/ae661bffe572880472249097c9b29c42b09650ea/city.csv
- published: 2021-10-11 — last commit touching this file: ae661bffe572880472249097c9b29c42b09650ea dated 2021-10-11T16:23:34Z (GitHub API, repo hflabs/city); pinned by SHA so the artefact cannot change under us
- license: CC BY-SA 4.0 (stated in repository README)
- downloaded: 2026-08-01T20:30:51Z
- SHA-256 (published file): `aefd11fd133881c348e0138b273a300ec6b3909c8f61d8d1d5ffc5e85021ec0d`
- Cyrillic headers: False; Cyrillic values: True
- duplicate rows: 0.0000; digits per row: 95.6; highest-entropy feature: `address` (100% unique)
- baselines: LR 0.740 (κ=0.48), GBT 0.750, majority 0.500 → lift +0.250
- **H3: admitted**
- serialisation variants and hashes:
  - `utf8_comma`: `f61808c0772a99b31a8f432de5ff8a8e…`
  - `utf8_semicolon`: `bbec2ce4b990db334d416c270ff74213…`
  - `cp1251_semicolon`: `d8fbd250eadba778fc8290839622696b…`
  - `utf8_semicolon_decimal_comma`: `e5db952f865728d1498ad311b0e0f0e8…`
- notes: 1117 Russian cities, DaData/HFLabs reference. 206 stars, 121 forks — the most widely reused of the candidates, so the strongest prior for having entered pretraining corpora. Modeling view: target 'Население выше медианы', dropped ['population'], constant columns ['country']

**`govdomains`** — 7528 rows × 19 columns

- source: https://raw.githubusercontent.com/infoculture/govdomains/f0cf2aa1360cdf121ca40dec8316fc069c6a570f/refined/feddomains.csv
- published: 2020-08-07 — last commit touching this file: f0cf2aa1360cdf121ca40dec8316fc069c6a570f dated 2020-08-07T15:41:39Z (GitHub API, repo infoculture/govdomains); pinned by SHA so the artefact cannot change under us
- license: Other (LICENSE.txt in repository)
- downloaded: 2026-08-01T20:31:01Z
- SHA-256 (published file): `565ee379b774ae88c42056a442ef64fb45c45aae1cd3e5bae609ee9d1d1b6817`
- Cyrillic headers: False; Cyrillic values: True
- duplicate rows: 0.0009; digits per row: 22.3; highest-entropy feature: `Domain` (100% unique)
- baselines: LR 0.930 (κ=0.85), GBT 0.931, majority 0.652 → lift +0.280
- **H3: admitted**
- serialisation variants and hashes:
  - `utf8_comma`: `7c12e7a0d5faf0ae85bbae6b8c8241b4…`
  - `utf8_semicolon`: `6c68a6407237cc6a85544a7dc1c3269b…`
  - `utf8_semicolon_decimal_comma`: `ad3bb8afc8a24b5a9df29cee2b4f32fe…`
- notes: Russian federal government domains, Infoculture. Modeling view: target 'Поддержка HTTPS', dropped ['HTTPS Support'], constant columns ['Is archived', 'Archives']

**`mos_zemelnye_uchastki`** — 5881 rows × 8 columns

- source: https://raw.githubusercontent.com/infoculture/mosopendata/ed80f2d28156f663cc337fb8935f8de91623a221/old/thedata/483_zemelnye_uchastki_v_sobstvennosti_goroda_moskvy.csv
- published: 2013-07-24 — last commit touching this file: ed80f2d28156f663cc337fb8935f8de91623a221 dated 2013-07-24T15:16:04Z (GitHub API, repo infoculture/mosopendata); pinned by SHA so the artefact cannot change under us
- license: not stated; mirror of data.mos.ru publications
- downloaded: 2026-08-01T20:31:25Z
- SHA-256 (published file): `4a8a2953762831b07407a05897085f397d79a6942b8d9260e851cc32c8cd54d9`
- Cyrillic headers: True; Cyrillic values: True
- duplicate rows: 0.0000; digits per row: 33.5; highest-entropy feature: `№ строки` (100% unique)
- baselines: LR 0.799 (κ=0.17), GBT 0.792, majority 0.775 → lift +0.024
- **H3: excluded — task not learnable; used for H1/H2 only**
- serialisation variants and hashes:
  - `utf8_comma`: `11b8ddce2d591dd64661a0bd5d3cb984…`
  - `utf8_semicolon`: `652f5a5c5d8e0578554ed939cdc9f9c4…`
  - `cp1251_semicolon`: `471e29957277ec39ee0087425eb5dc0c…`
  - `utf8_semicolon_decimal_comma`: `f1cb14dae5fe60bef659bee7e67ef9b4…`
- notes: Moscow city land plots. The only verified source with Cyrillic column names as well as values. Tab-separated despite the .csv extension. Repository frozen since 2014. Modeling view: target 'Аренда', dropped ['№ строки', 'Идентификатор', 'Тип собственника']

**`mos_torgovye_obekty`** — 10212 rows × 11 columns

- source: https://raw.githubusercontent.com/infoculture/mosopendata/ed80f2d28156f663cc337fb8935f8de91623a221/old/thedata/486_nestatsionarnye_torgovye_obekty.csv
- published: 2013-07-24 — last commit touching this file: ed80f2d28156f663cc337fb8935f8de91623a221 dated 2013-07-24T15:16:04Z (GitHub API, repo infoculture/mosopendata); pinned by SHA so the artefact cannot change under us
- license: not stated; mirror of data.mos.ru publications
- downloaded: 2026-08-01T20:31:37Z
- SHA-256 (published file): `4c3335ed3a604df1f9a935668c7c178fa573848a98972055d7aa3d789f2a68b0`
- Cyrillic headers: True; Cyrillic values: True
- duplicate rows: 0.0000; digits per row: 53.7; highest-entropy feature: `№ строки` (100% unique)
- baselines: LR 0.818 (κ=0.56), GBT 0.816, majority 0.642 → lift +0.176
- **H3: admitted**
- serialisation variants and hashes:
  - `utf8_comma`: `f5109a6046ba4e8dce2b83a341482986…`
  - `utf8_semicolon`: `48befb1c8bfbda119518429c608e2fc3…`
  - `cp1251_semicolon`: `5f8c4285bb0a599c70e7208577a9c5c9…`
  - `utf8_semicolon_decimal_comma`: `48befb1c8bfbda119518429c608e2fc3…`
- notes: Moscow non-stationary retail objects, 10212 rows. Cyrillic column names and values; tab-separated despite the .csv extension. Modeling view: target 'Модульный объект', dropped ['№ строки', 'Номер свидетельства', 'Дата свидетельства', 'Вид нестационарного объекта']

**`russian_retail`** — 2737 rows × 11 columns

- source: https://www.kaggle.com/datasets/pavelkunitsyn/russian-retail
- published: 2021-08-11 — Kaggle API lastUpdated=2021-08-11T18:49:36Z; file mtime inside the downloaded archive 2021-08-11 18:49:38
- license: Other (specified in description); unclear — flagged in the amendment
- downloaded: 2026-08-01T20:32:36Z
- SHA-256 (published file): `da0d2a22751739a59542cc278b2c5aeb821b3e41f24602ac2a633fb7e211f745`
- Cyrillic headers: False; Cyrillic values: True
- duplicate rows: 0.0026; digits per row: 20.4; highest-entropy feature: `description` (100% unique)
- baselines: LR 0.678 (κ=0.12), GBT 0.677, majority 0.659 → lift +0.018
- **H3: excluded — task not learnable; used for H1/H2 only**
- serialisation variants and hashes:
  - `utf8_comma`: `b2a969a321697e98bb464fdde4d02266…`
  - `utf8_semicolon`: `90069e0992e958306146fce082ccc96e…`
  - `utf8_semicolon_decimal_comma`: `871c441c48e04ed0ebdac7348cdf4344…`
- notes: Russian retail brands from Kaggle, 2737 rows. English headers, Cyrillic values. The free-text 'description' column is dropped from the modeling view (it is prose, not a tabular feature) but stays in the published file, where it is the highest-entropy field for the completion tests.

### Fresh control, collected after 2026-06-01

**`trudvsem_vacancies_2026`** — 1196 rows × 13 columns

- source: http://opendata.trudvsem.ru/api/v1/vacancies
- published: 2026-06-01/2026-08-01 — constructed by us from vacancy records whose creation-date field is on or after 2026-06-01; API queried 2026-08-01T19:52:36Z; span ['2026-06-01', '2026-08-01']
- license: Russian federal open data (Работа России), Методрекомендации Минцифры v3.0; open API without key
- downloaded: 2026-08-01T19:53:35Z
- SHA-256 (published file): `5f98ebfad75923382c48660e9137e5da50c1911429c52f65e90ec5866e2d6911`
- Cyrillic headers: True; Cyrillic values: True
- duplicate rows: 0.0000; digits per row: 4.8; highest-entropy feature: `Должность` (70% unique)
- baselines: LR 0.839 (κ=0.68), GBT 0.810, majority 0.536 → lift +0.303
- **H3: admitted**
- serialisation variants and hashes:
  - `utf8_comma`: `5f98ebfad75923382c48660e9137e5da…`
  - `utf8_semicolon`: `ca5829e3843883d44fe5f8a93d0ac1f6…`
  - `cp1251_semicolon`: `6d3735cbe520f2f68136cbbc4c769288…`
  - `utf8_semicolon_decimal_comma`: `ca5829e3843883d44fe5f8a93d0ac1f6…`
- notes: Fresh control / modern twin of Adult: predict salary above median from job attributes. Salary-bearing fields (salary, salary_min, salary_max, duty) dropped; automated leakage guard passed. LR 0.839 / GBT 0.810 vs majority 0.536.

### Western canon (reference, unchanged from Bordt et al.)

**`iris`** — 150 rows × 5 columns

- source: https://archive.ics.uci.edu/dataset/53/iris
- published: 1988-07-01 — UCI ML Repository donation date; file byte-identical to the copy shipped with tabmemcheck and with the COLM paper code (md5 bab8c78b)
- license: CC BY 4.0 (UCI)
- downloaded: 2026-08-01T19:18:07Z
- SHA-256 (published file): `9194e2b71f7144e7d192a1c38f9a54f26b0e8f705c0929b8225b0cd10275efd1`
- Cyrillic headers: False; Cyrillic values: False
- duplicate rows: 0.0200; digits per row: 8.0; highest-entropy feature: `petal_length` (29% unique)
- serialisation variants and hashes:
  - `utf8_comma`: `20f7ef9ad6e85c0752a0cda4c9d1edfc…`
  - `utf8_semicolon`: `469156299c5b5c6cff0090a3a9239cbc…`
  - `cp1251_semicolon`: `bf5b4d457da721e72495f3889f3ded45…`
  - `utf8_semicolon_decimal_comma`: `bf5b4d457da721e72495f3889f3ded45…`
- notes: Western canon control; reproduced in RESULTS_GATE.md

## Canonicalisation

Verbatim memorization is a property of the exact bytes a model saw, and these files have no single canonical form: the same table circulates as cp1251 and UTF-8, comma- and semicolon-separated, with `.` or `,` as the decimal mark. Testing one guessed form risks reporting absence of memorization when we merely guessed wrong. Every dataset is therefore materialised in each serialisation listed above, all of them are tested, results are reported per variant, and the maximum across variants is the extraction estimate. This has no counterpart in Bordt et al., whose canon has one unambiguous form per dataset.

## Deviations from §4 as written

- §4 set a target of at least four Russian pre-cutoff datasets with a minimum of two. Five are frozen here, of which three carry a learnable task and enter H3.
- §4 named Kaggle, data.gov.ru, data.mos.ru and Rosstat as candidate sources. data.gov.ru is now an empty single-page application with no catalogue or API, rosstat.gov.ru did not respond, and data.mos.ru serves data only through an API requiring a key. Two datasets here come from a 2013-2014 GitHub mirror of data.mos.ru instead, which has the side benefit of a provable publication date.
- One dataset (`russian_retail`) comes from Kaggle under an unclear licence ('Other, specified in description'). It is used for memorization testing only, no redistribution, and this is flagged in the paper's data statement.
