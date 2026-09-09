# Amendment 7 to PREREGISTRATION.md — the near-duplicate half of the row-completion baseline

**Date:** 2026-09-09. **Scope:** the decision rule of the row-completion test in
§5, whose baseline is written as "the dataset's duplicate/near-duplicate base
rate". Only the duplicate half of that clause was ever implemented. This
amendment implements the other half, repairs the null it left degenerate,
names the columns that can witness memorization on each dataset, excludes
queries that are not rows, and restates the fresh-twin precondition of
`AMENDMENT_5` §4 for registry-style data. No hypothesis, confirmation
threshold, α, correction procedure, model or dataset changes.

**Transparency.** The rule is written after session C1 was scored. The
numbers that forced it are in `LOG.md` 2026-09-09 and in §4 below; they were
seen first. The rule is decided on the argument in §1, which is the argument
the test's authors make for the test's validity, and every alternative that
was considered is listed in §5 with the measurement that rejected it. The
iris anchor, the only cell with outside ground truth, is positive under every
rule below and under none of the rejected ones is it treated worse than the
cells the rule is meant to catch.

---

## 1. Why the duplicate rate is not the baseline the test needs

The row-completion test shows a model eight consecutive rows of a file and
asks for the ninth. Its authors state the assumption it rests on: tabular
rows "contain random variables, and it is impossible to consistently
reproduce the realizations of random variables unless the values of the
random variables have been seen before" (`tabmemcheck` README), and "one
assumption is that the dataset has non-zero entropy ... we also assume ...
that the rows in this csv file are ordered at random" (Bordt et al., arXiv:2403.06644 §3).
They score the count against the file's duplicate rate, which covers one way
of being right without memory — hitting a repeated row — and they note the
other: "the Iris dataset contains many rows that are near-duplicates. This
means that an LLM might also achieve a non-zero row completion rate by chance
or prediction. Because of this complication, we manually judge the results"
(arXiv:2404.06209, Supplement D). §5 of our preregistration replaced their
manual judgement with a formal rule and wrote "duplicate/near-duplicate base
rate" to carry both halves. The second half was never operationalised because
no canon file needed it.

The Russian files do. A registry export sorted by organisation carries runs
of rows that differ only in a counter or a region name (`deti.r36.nalog.ru`
→ `deti.r37.nalog.ru`; `7791 Садко трейдинг …` → `7792 Садко трейдинг …`).
The ninth row is then largely determined by the eight before it, and what
remains is filled from world knowledge (region code 12 is Марий Эл). On
govdomains 54% of all eight-row windows have a ninth row within normalised
edit distance 0.1 of one of the eight; on the canon the share is 0–1.7%
except iris (27%), which the authors flagged.

Two consequences follow and both are repaired here. First, on a file with
no exact duplicate the null of the binomial test was degenerate: the runner
tested against ε = 10⁻⁹, so a single match in 250 queries returned
p = 2.5 × 10⁻⁷ whatever the file. Second, the rate-against-own-baseline
comparison that `AMENDMENT_5` §1 requires across datasets is not a comparison
when the share of prompt-determined rows differs between files by a factor
of fifty.

---

## 2. The rule

**R1 — near-duplicate queries are not counted.** A query is a near-duplicate
of its own context when the true ninth row lies within normalised
Levenshtein distance τ = 0.10 of at least one of the eight prompt rows
(distance divided by the longer length). Such queries are removed from both
the count and the denominator. The condition depends only on the prompt and
the true row, never on the model's answer, so it conditions on the design
and not on the outcome. The share of near-duplicate windows in the whole
file is reported beside digits-per-row as a second entropy covariate. The
threshold is the complement of the edit-similarity thresholds used to define
near-duplicates in the deduplication literature — 0.8 in Lee et al. (arXiv:2107.06499)
and 0.7 in Zhang et al. (arXiv:2112.12938), i.e. τ = 0.2 and 0.3 — and is the
strictest of the three, the one that removes the fewest queries. Because
those thresholds were set for web documents and not for thirty-character
CSV rows, every reported cell carries the count at τ ∈ {0.05, 0.10, 0.20, 0.30};
a verdict that is not stable across that range is reported as
threshold-dependent.

**R2 — the null is never zero.** The one-sided exact binomial test is
against p₀ = max(d, q, 3/W), where d is the file's duplicate-row rate as
before, q is the exact-hit rate over all eight-row windows of the file of a
predictor that sees only the prompt (copy the last row; or, when the last two
rows differ only in digit runs, advance every changed run by the same step),
and 3/W, with W the number of windows, is the 95% upper bound on a rate
observed to be zero in W trials (Hanley & Lippman-Hand, JAMA 1983, the rule
of three). The predictor is the row-completion form of the classifier the
authors prescribe for non-random rows in the first-token test
(arXiv:2403.06644 §3) and is reported even where it does not bind. ε is
removed from the runner and from `src/detectability.py`; the minimum
detectable rates of `AMENDMENT_5` §1 are recomputed under this null.

**R3 — a positive verdict needs a witness.** For every dataset, §3 names the
columns whose values are realisations of high-entropy variables that no
public source outside the file supplies. A row-completion cell can be
positive only if, over its queries, the model reproduced verbatim, as a
complete field of the scored line, at least one witness value that occurs
nowhere in the prompt. The number of such reproductions, and the number of
witness values that were available, are reported for every cell whether or
not it is positive, together with the same counts for the non-witness
columns — the two-tier report §7 requires. This is the authors' own validity
argument applied at field level, and it needs no threshold.

**R4 — a query whose target is not a row is not a query.** `tabmemcheck`
reads a CSV as physical lines. Where a file carries multi-line quoted
fields, some lines are fragments of a record. A query whose true "row" does
not parse into the header's number of fields is excluded from count and
denominator and reported. Of the twelve frozen files only govdomains is
affected (243 continuation lines and 168 malformed lines among 7,801; 15 of
the 250 queries of session C1).

**R5 — the fresh twin decides registry-style cells.** `AMENDMENT_5` §4
already requires a construction-matched fresh control before H2 verdicts.
It is restated here as a precondition specific to R1–R4: a positive
row-completion verdict on a Russian dataset whose near-duplicate share
exceeds 10% is not stated until the fresh twin of that construction has been
run on the same plan and is negative under R1–R4. The twin's rate is the
empirical value of "what a model that never saw the file produces on a file
of this shape", which no model-free predictor bounds from above.

The header test is unchanged: its 500-character prompt contains no run of
templated rows to continue, and it failed on every Russian file in C1. The
feature-completion and first-token tests keep their own baselines (§5,
`AMENDMENT_6` §1).

---

## 3. Witness columns

Chosen on one principle: a value counts as a witness when it is a
measurement, coordinate, identifier or amount that no reference book or
naming convention reproduces. Public classifier codes (KLADR, OKATO, OKTMO,
tax-office codes, postal codes), sequential ids, dates that repeat down a
template, and names that follow the file's own pattern are not witnesses.

| dataset | witness columns | excluded although numeric, and why |
|---|---|---|
| iris | sepal_length, sepal_width, petal_length, petal_width | — |
| uci-wine | the thirteen chemistry columns | target (class label) |
| openml-diabetes | Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction | Pregnancies, Age, Outcome (low entropy) |
| adult-train | fnlwgt | every other column is a category or a small integer |
| california-housing | longitude, latitude, total_rooms, total_bedrooms, population, households, median_income, median_house_value | housing_median_age (low entropy) |
| titanic-train | Ticket, Fare, Cabin | PassengerId (sequential); Name (famous passengers are world knowledge) |
| govdomains | IPs, ASN | Statuscode (three values); everything else is a name or a category |
| hflabs_city | fias_id, geo_lat, geo_lon, population | kladr_id, okato, oktmo, tax_office, postal_code (public classifiers); foundation_year (encyclopaedic) |
| mos_zemelnye_uchastki | Кадастровый номер, Площадь (кв.м) | № строки (sequential); Идентификатор (format under review) |
| mos_torgovye_obekty | Номер свидетельства, Дата свидетельства | № строки (sequential); contract dates (repeat down the template) |
| russian_retail | total_rented_area, presence_russia | founded (encyclopaedic); presence_world (small integers) |
| trudvsem_vacancies_2026 | none needed: the fresh control must be zero under every rule | — |

The hflabs_city split was drawn after session C1 showed the two model
reproducing 10–23% of KLADR/OKATO/OKTMO codes absent from the prompt and 0
of 250 FIAS ids, coordinates or population figures. The principle that a
published classifier is world knowledge and a GUID is not does not depend on
that observation, but the observation is what made the distinction visible,
and the record says so.

---

## 4. Session C1 under the rule

Computed by `src/prefix_baseline.py` from the call logs
(`results/prefix_baseline_ru_probe_20260908T121529Z.json`). Counts by the
library's criterion, then over the queries R1 (τ = 0.10) and R4 keep, the
p-value against the R2 null, and the witness count of R3; base =
Mistral-Nemo, adapted = Vikhr-Nemo.

| dataset | model | library | R1 + R4 | p (R2) | witness values absent from prompt | reproduced | verdict |
|---|---|---|---|---|---|---|---|
| iris | base | 51/142 | 36/103 | 9 × 10⁻³⁴ | 247 | 151 | positive |
| iris | adapted | 32/142 | 24/103 | 2 × 10⁻¹⁸ | 247 | 130 | positive |
| govdomains | base | 32/250 | 2/111 | 0.73 | 47 | 0 | negative |
| govdomains | adapted | 19/250 | 1/111 | 0.93 | 47 | 0 | negative |
| mos_torgovye_obekty | both | 1/250 | 0/234 | 1 | 394 | 0 | negative |
| hflabs_city | both | 0/250 | 0/250 | 1 | 999 | 0 | negative |
| mos_zemelnye_uchastki | both | 0/250 | 0/249 | 1 | 493 | 0 | negative |
| trudvsem (fresh) | both | 0/250 | 0/207 | 1 | — | — | zero, gate holds |

The R2 null is 0.023 on govdomains (the predictor rate binds), 1.1 × 10⁻³
on mos_torgovye_obekty (predictor), 2.7 × 10⁻³ on hflabs_city (rule of
three), 0.021 on iris (rule of three, just above its duplicate rate). The
τ-curve for govdomains after R4 is 6/157, 2/111, 0/57, 0/27 across τ = 0.05,
0.10, 0.20, 0.30 for the base model and 3/157, 1/111, 0/57, 0/27 for the
adaptation; for iris the count is positive at every τ while the denominator
shrinks to 9 at τ = 0.20, which is why R3 and not R1 carries the verdict.
On the queries the rules keep, the paired contrast on iris is base 19 /
adapted 7 discordant, McNemar p = 0.029 (29 / 10, p = 0.0034 over all
queries): the direction of block B survives the rule.

---

## 5. Alternatives considered and rejected

- *The letter of §5, duplicate rate only.* Produces a positive verdict on
  one match against ε and contradicts the clause's own second half.
- *The file's near-duplicate share as the null.* Tests Vikhr-Nemo on iris
  at 32/142 against 0.268, p = 0.89, on a cell where the header test passes
  and the first-token test gives p = 1.8 × 10⁻¹¹; the quantity is the share
  of easy queries, not the chance of answering one.
- *The library's whole-file `row_independence_test` as a gate for row
  completion.* Rejects iris (sorted by species), wine, titanic and
  california-housing on every seed (`results/row_independence_whole_file.json`);
  the authors wire it only to the first-token test.
- *Per-row compressibility or entropy filters.* Reported not to track
  memorization at the level of single examples (Huang et al., arXiv:2507.06056).
- *A negative-control model.* No available model is known not to have seen
  Russian registries.
- *Changing the probe* (non-contiguous prompt rows; choice among perturbed
  rows). Sound, but a different instrument; loses comparability with the
  authors' tables. Noted for the limitations section.

---

## 6. What does not change

Hypotheses, confirmation and refutation thresholds, α and the Holm
correction within families, the four tests and their prompts, the datasets
and their hashes, the primary measure (exact match) and the reporting rules
of `AMENDMENT_5`. The near-match rate of `AMENDMENT_3` §2 stays descriptive.
