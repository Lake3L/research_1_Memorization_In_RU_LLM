# Amendment 5 to PREREGISTRATION.md — how a zero is reported, and which measure decides

**Date:** 2026-08-15. **Scope:** reporting requirements, one measurement
hierarchy, and one addition to the control set. No hypothesis, threshold,
correction procedure or dataset already frozen changes. §7 — never claiming
memorization from knowledge-level signals — is reinforced here, not relaxed.

Two things forced this, and both were found by checking an assumption rather than
by new data. The detectability calculation showed that our nulls are much stronger
than they look, but that the datasets differ enormously in how hard they are to
reproduce at all. And the graded measure introduced in `AMENDMENT_3` §2 turned
out, when actually tested against the binary one, to disagree with it in
direction. Both have to be settled in writing before block C produces the numbers
they govern.

---

## 1. Every zero is reported with the smallest effect it could have excluded

**The rule.** No count of zero is reported on its own. It carries the minimum
detectable rate for that cell — the smallest true memorization rate a one-sided
exact binomial test would have rejected the dataset's own duplicate baseline at,
given the queries actually run — and the dataset's digits-per-row.

**Why.** "The model did not memorize this dataset" and "our test could not have
told us either way" are different claims, and a bare zero does not distinguish
them. For the canon we can lean on outside ground truth: Bordt et al. established
that GPT-4 reproduces iris, and our own reproduction confirmed it. For
`mos_torgovye_obekty` nobody knows what the answer should be. That is precisely
why the dataset is worth testing, and precisely why a null on it means nothing
until its sensitivity is stated next to it.

Computed by `src/detectability.py`, at 250 queries:

| dataset | duplicate baseline | min detectable rate | digits per row |
|---|---|---|---|
| iris | 0.0200 | 6.3% | 8.0 |
| uci-wine | 0.0000 | 0.9% | 38.9 |
| openml-diabetes | 0.0000 | 0.6% | 19.2 |
| hflabs_city | 0.0000 | 0.6% | **95.6** |
| mos_torgovye_obekty | 0.0000 | 0.6% | 53.7 |
| govdomains | 0.0009 | 1.2% | 22.3 |
| russian_retail | 0.0026 | 1.7% | 20.4 |

**What this buys.** These nulls are strong. A zero at 250 queries on
`mos_torgovye_obekty` excludes memorization above 0.6%. That is a real finding and
can be stated as one.

**What it does not buy, and this is the part that must travel with it.** The 0.6%
is *statistical* sensitivity. It says nothing about how hard the row is to
reproduce. A `hflabs_city` row carries 95.6 digits against iris's 8.0, and Ward et
al. show that verbatim reproduction degrades with the length of the digit string,
saturating around 100 digits. The same rate on those two datasets is not the same
feat. Consequently:

- comparisons are always **rate against that dataset's own baseline**, never raw
  counts, and never counts compared across datasets;
- digits per row is reported beside every such comparison, as Ward's covariate;
- a Russian dataset's null is never presented as equivalent evidence to a canon
  dataset's null without that covariate in view.

---

## 2. Block C sessions carry an in-session anchor

**The rule.** Every run on Russian datasets includes iris, in the same session, on
the same model, at the same settings. Its result is reported alongside.

**Why.** A zero on a Russian dataset has three possible causes: the model never
saw the file, the test cannot extract from that model at all, or the session was
broken in some way the mock controls do not catch. The mock controls test our
counting code; they say nothing about whether *this model* can be made to
reproduce anything. Iris is the one dataset where we have external ground truth
that the data sits in the pretraining corpora of the model families we test, and
where our own pipeline has extracted it.

**What the anchor proves, stated narrowly so it is not over-read.** It does *not*
establish sensitivity on a Russian dataset — the entropy argument in §1 forbids
that transfer. It establishes that the model, the pipeline and the session were
jointly capable of extracting something. A Russian zero in a session where iris
also came out zero is uninterpretable and is reported as such. A Russian zero in a
session where iris fired is evidence.

---

## 3. The measurement hierarchy: exact-match rate decides, the graded measure describes

**The rule.**

- **Primary, for H1, H1b, H2 and H4: the exact-match rate, byte-exact.** No number
  normalisation, no near-match credit. This is the quantity the hypotheses are
  about and the only one from which a verbatim claim may be made.
- **Secondary: the normalised edit distance**, reported *with its distribution* —
  the count of answers that were exact, near (≤0.1), partial and unrelated — never
  as a mean alone.
- For **H3**, the predictor of "seen" status is the exact-match rate. The graded
  measure enters only as a robustness check.

**Why, and this revises the impression left by `AMENDMENT_3` §2.** That amendment
introduced the graded measure on the argument that five identical zeros in the
primary outcome can be five different numbers underneath. The argument is sound
and the measure stays. But it was written before the measure had been compared
against the binary one on real data, and it left the impression that the graded
quantity could carry the paired analysis. Tested on the 12B pilot
(`src/compare_measures.py`, iris row completion, n=50 per arm), it cannot:

| | base (Mistral-Nemo) | adapted (Vikhr-Nemo) | p |
|---|---|---|---|
| exact matches | 4/50 = 0.080 | 7/50 = 0.140 | 0.525 |
| mean normalised distance | **0.195** | 0.240 | 0.735 |

The two measures point in opposite directions — the binary favours the adapted
model, the graded favours the base — and neither is close to significant. The
distribution shows why:

| band | base | adapted |
|---|---|---|
| exact | 5 | 7 |
| near (≤0.1) | 5 | 4 |
| partial | 40 | 31 |
| unrelated (>0.5) | **0** | **8** |

The adapted model is bimodal: right more often *and* wrong more badly. A mean over
that summarises nothing, which is why the distribution is now mandatory and the
mean alone is not admissible. The exact count differs by one between the two rows
above — 4 byte-exact against 5 after canonicalisation — and that difference is
itself the reason the primary measure stays byte-exact: normalisation turns a
formatting difference into a match, which is acceptable for describing closeness
and not acceptable for claiming reproduction.

**The boundary, restated because it is the easiest one to cross by accident.** A
small edit distance is not memorization. On titanic the model returned a row with
the fare `7.8958` exactly right, which reads as recall until it is counted:
`7.8958` is the third most common fare in the file, 38 rows of 891, and 47 tickets
share the `3492` prefix that the model nearly matched. That is knowledge of the
distribution. §7 forbids reporting it as memory, and the hierarchy above is how
that prohibition is made operational rather than merely stated.

---

## 4. A second fresh control, construction-matched

**The rule.** Before H2 and H3 verdicts, a second fresh control is collected: a
dataset built to mirror the schema of one of the five Russian pre-cutoff datasets,
collected after 2026-06-01, with a provable collection date and comparable digits
per row.

**Why.** The fresh controls are the only hard ground truth in the whole design.
"Seen" is always inferred; "unseen" is certain, because we collected the data
ourselves after every model cutoff. That asymmetry is what the H3 contrast rests
on, and it is what makes the §4 validity gate meaningful — a positive verbatim
verdict on fresh data invalidates that model×test cell.

One fresh control is not enough for that job. `trudvsem_vacancies_2026` is
vacancies, with 4.8 digits per row, the lowest of any dataset we hold. If a model
comes out zero on it, we cannot separate "the model has not seen it" from "rows of
this shape are easy and it still produced nothing", nor from "this domain is
unlike the pre-cutoff datasets it is being contrasted with". A twin matched to a
pre-cutoff dataset's schema removes the domain and difficulty differences from the
contrast, which is exactly the move Bordt et al. made by pairing Adult with ACS
Income.

**Fixed here so the choice cannot be made to fit a result:** the twin is chosen
and registered — source, schema, collection date, hash — *before* any model is run
on it, and the pre-cutoff dataset it mirrors is named at the same time.

---

## 5. What does not change

Hypotheses, confirmation and refutation thresholds, the Holm correction within
each family, the four verbatim tests and their criteria in §5, the dataset freeze
of Amendment 1 as corrected by Amendment 2, and the confound plan of §7. This
amendment governs how results are reported and which measure carries a claim; it
does not move any line a claim has to cross.
