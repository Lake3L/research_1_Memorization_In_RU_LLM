# Memorization of Tabular Data in Russian-Language LLMs

Do Russian-centric and Russian-adapted LLMs (T-lite/T-pro, Vikhr, ruadapt) memorize
(a) the Western canon of tabular datasets (Adult, Titanic, California Housing, Iris, Wine, Diabetes) and
(b) open Russian-language tabular datasets — and how does this contamination distort the evaluation of their few-shot abilities in Russian?

Builds on the memorization tests of Bordt et al. (COLM 2024, [arXiv:2404.06209](https://arxiv.org/abs/2404.06209)) and the [`tabmemcheck`](https://github.com/interpretml/LLM-Tabular-Memorization-Checker) tool, adapted to HuggingFace models and Russian-language prompts.

## Hypotheses

Preregistered before any experiments — see [PREREGISTRATION.md](PREREGISTRATION.md).

- **H1.** Russian-centric models memorized the Western tabular canon (their pretraining includes the English web).
- **H2.** Russian-centric models additionally memorized open Russian tabular datasets unknown to multilingual controls.
- **H3.** Memorization inflates few-shot performance: the gap between seen and post-cutoff datasets is significant.
- **H4.** Memorization test outcomes depend on the prompt language (Russian vs English) for the same model.

## Repository layout

```
NOVELTY_CHECK.md    novelty protocol results (search queries, findings, verdict)
PREREGISTRATION.md  frozen hypotheses, metrics, and decision criteria
LOG.md              lab journal
paper.md            paper draft, grown incrementally
src/                adapted tests and experiment code
data/               dataset collection scripts (with collection dates)
results/            experiment outputs backing the paper's numbers
```

## Commit policy

The history is part of the record and is written for a reviewer.

- The subject says what changed in the record — a script, a result file, a frozen
  dataset, an amendment. Imperative, English, at most 72 characters.
- A body appears only when the diff cannot be read without a fact: the number, the
  rule, the file it comes from. A few lines at most. No process narration, no
  pending decisions, no account of who asked for what — that belongs in `LOG.md`
  and in the dated amendments.
- One unit of work per commit; a result and the script that produced it travel
  together.
- No attribution trailers. The repository owner is the author of record; tools are
  disclosed in the paper, not in the history.
- Frozen documents are never rewritten in place; a change is a new dated amendment.

## Reproducing

Will be a single command from a clean clone; exact instructions appear here together with the first committed results. All reported numbers use ≥3 seeds (mean ± std), fixed model revisions (HF commit hashes), and pinned dependency versions.
