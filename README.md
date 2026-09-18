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

## Reproducing

Will be a single command from a clean clone; exact instructions appear here together with the first committed results. All reported numbers use ≥3 seeds (mean ± std), fixed model revisions (HF commit hashes), and pinned dependency versions.
