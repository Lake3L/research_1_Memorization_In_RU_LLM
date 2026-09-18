# Amendment 3 to PREREGISTRATION.md — a graded secondary outcome for H1/H1b, and the branch for block B

**Date:** 2026-08-11. **Written before any block B data exists.** The only run
that has informed it is the §8 gate, which is a validation of the instrument and
not a test of any hypothesis; using it to calibrate the instruments is what a gate
is for. No hypothesis, no confirmation threshold and no correction procedure in
PREREGISTRATION.md §5–§6 is weakened by this amendment; §6 (H1b) gains a secondary
outcome, and §8's own criterion is untouched.

## 1. Why

The gate (RESULTS_GATE.md §6) found extractable verbatim memorization in
`Qwen/Qwen2.5-7B-Instruct` on iris and on nothing else: row completion 13/50 on
iris, 0 on the other five canon datasets, feature completion 0/50 everywhere,
header passing on iris alone.

`Qwen2.5-7B-Instruct` is the base member of two of the three base↔adapted pairs.
H1b asks whether Russian adaptation retains, attenuates or amplifies inherited
memorization. On five of six datasets there is nothing in the base to retain,
attenuate or amplify, and a paired comparison of zero against zero has no power
whatever the adaptation did. This was foreseen — the reading notes for Bordt's
Table 3 record that at the 7-8B scale extractability is lower than GPT-4's and
that zeros threaten us for reasons other than absence of memorization — and it is
now measured rather than feared.

The exact-match counts are not, however, the only thing in the data. The gate also
recorded, per cell, the normalized Levenshtein distance between the true next row
and the model's answer:

| dataset | row completion | mean normalized Levenshtein | near-match (≤0.1) |
|---|---|---|---|
| iris | 13/50 | 0.16 | 38% |
| california-housing | 0/25 | 0.38 | 0% |
| adult-train | 0/25 | 0.42 | 4% |
| uci-wine | 0/25 | 0.44 | 0% |
| openml-diabetes | 0/25 | 0.47 | 0% |
| titanic-train | 0/25 | 0.50 | 0% |

Five identical zeros in the primary outcome are five different numbers here. A
model that returns a row with one digit changed has not failed in the same way as
one that returns an unrelated row, and the binary verdict discards that.

(These distances are already computed under the canonicalisation rule defined
below, by `src/rescore_calls.py`. The values written by the gate run itself
predate the rule and differ slightly; the counts do not differ at all, because
canonicalisation never touches the verbatim measure.)

## 2. What is added

**Secondary outcome for H1 and H1b, confirmatory in direction, never in kind.**

- **Quantity.** Mean normalized Levenshtein distance
  `d = lev(truth, answer) / max(|truth|, |answer|)` between the true next row and
  the first non-empty line of the response, averaged over the queries of a
  (model × dataset × variant × prompt-language) cell. Lower is closer to verbatim.
- **Normalization before comparison.** Number formats are normalized first, per
  Ward et al.'s caveat and PREREGISTRATION.md §6 (H4): otherwise a decimal-comma
  artefact is scored as a memorization difference. This matters for H4 and for the
  Russian datasets, not for the English canon, but the rule is the same everywhere.
- **Test.** The same paired structure as the primary H1b outcome: base vs adapted
  per dataset, Wilcoxon across datasets, Holm within the H1 family.
- **Companion descriptives.** The near-match rate at `d ≤ 0.1` is reported
  alongside, as a descriptive quantity only. The threshold is our choice and has
  no counterpart in Ward et al., who use the distance as a continuous score with a
  threshold read off the score distribution; the confirmatory quantity is
  therefore the continuous mean, not the thresholded rate.
- **Covariate.** Digits per row is reported with every such comparison (Ward: the
  length of digit strings governs reproducibility, so counts are not comparable
  across datasets without it). It is already in the registry diagnostics.

**What this is not.** A small Levenshtein distance is not evidence of verbatim
memorization and may never be reported as such. PREREGISTRATION.md §7 stands
unchanged: verbatim claims rest on the four verbatim tests. The secondary outcome
exists to give the *paired* comparison a graded quantity, and to distinguish a
model that nearly reproduces a row from one that does not — a distinction that
matters precisely because it is *not* the same as memorization.

## 3. Chat templates are not identical within a pair

Probed at load and recorded in every results file (`chat_template` field,
`src/hf_llm.py::chat_template_report`):

| model | system prompt is placed |
|---|---|
| `Qwen/Qwen2.5-7B-Instruct` | before the first user turn |
| `Vikhrmodels/Vikhr-Nemo-12B-Instruct-R-21-09-24` | before the first user turn |
| `mistralai/Mistral-Nemo-Instruct-2407` | **moved to immediately before the last user turn** |

Mistral's template relocates the system message, silently and without raising, so
the merge fallback in our backend never fires and nothing in the counts would
reveal it. In the Vikhr-Nemo ↔ Mistral-Nemo pair the adaptation therefore ships a
different template from its base, and the two arms differ in where the instruction
sits as well as in their weights.

**Decision: each model is queried through its own template.** Forcing a common
template would mean querying at least one model through an interface it was not
built for, which is a larger distortion than the one it removes, and it is not
what a user of these models would do. The asymmetry is instead made auditable —
probed, logged, and reported in the paper's limitations — and it is a candidate
explanation to be considered before any Vikhr-vs-Mistral difference is attributed
to Russian adaptation.

## 4. The branch for block B, committed in advance

One diagnostic run before the rest of the compute: the **Vikhr-Nemo-12B ↔
Mistral-Nemo-Instruct-12B pair**, same plan as the gate (canon, `raw`, English).
Both are apache-2.0, both are pinned in `models.lock`, neither is gated. The
question it answers is whether the floor observed at 7B is a scale effect.

- **If at least one 12B model passes the header test on ≥2 canon datasets, or
  shows a significant row-completion count on ≥2:** the floor is a scale effect,
  block B proceeds exactly as preregistered, and the 7B pairs are reported with
  their floor as part of the scale story.
- **If both 12B models floor as the 7B did:** the primary H1b comparison is
  reported as a null with its floor stated, the continuous secondary outcome of §2
  carries the paired analysis, and the paper's H1 contribution becomes the scale
  boundary of extractable tabular memorization in this model class — a negative
  result, published as §10 requires.

Either way the surface for H2 (Russian datasets, block C) is unchanged: those
datasets are not in the canon and their memorization is an independent question.
