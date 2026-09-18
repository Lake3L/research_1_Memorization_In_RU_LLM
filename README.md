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
amendments/         dated amendments to the preregistration, never rewritten in place
reports/            results documents, one per experiment block
LOG.md              lab journal
TODO.md             roadmap, one block per session
paper.md            paper draft, grown incrementally
src/                adapted tests and experiment code
data/               dataset registry with provenance and hashes; files are refetched by script
notebooks/          the GPU session driver and the session definition it reads
results/            raw call logs and per-run outputs backing every reported number
```

Every new amendment goes to `amendments/`, every new results document to `reports/`;
the root holds only the documents that frame the study.

## Reproducing

Will be a single command from a clean clone; exact instructions appear here together with the first committed results. All reported numbers use ≥3 seeds (mean ± std), fixed model revisions (HF commit hashes), and pinned dependency versions.
