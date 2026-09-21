# Block C, exposure sessions — the first positive Russian cells

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

## 6. Files

Raw logs `results/calls_exposure_1_*_20260920T174835Z.jsonl`, per-cell results
`results/gateA_exposure_1_*_20260920T174835Z.json`, rule-level scoring
`results/prefix_baseline_exposure_1_20260920T174835Z.json`. Reproduce the
tables with:

```
python src/report_run.py "results/gateA_exposure_1_*20260920T174835Z.json"
python src/rescore_calls.py results/calls_exposure_1_<model>_*.jsonl --results results/gateA_exposure_1_<model>_*.json
python src/prefix_baseline.py results/calls_exposure_1_*.jsonl --out results/prefix_baseline_exposure_1_20260920T174835Z.json
```

One defect was found and fixed while scoring this session:
`src/prefix_baseline.py` took its dataset groups from a hard-coded default
that predated the `ru_exposure` group, so the new cells were silently not
scored. The default is now every group in the registry, and a logged cell that
is not scored prints a line saying so.
