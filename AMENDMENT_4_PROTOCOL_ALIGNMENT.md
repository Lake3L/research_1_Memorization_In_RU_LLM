# Amendment 4 to PREREGISTRATION.md — align the protocol with the reference, and probe the prompting mode

**Date:** 2026-08-14. **Scope:** the parameters the memorization tests are run
with, and one added preregistered comparison. No hypothesis, decision rule,
dataset or correction procedure changes. The earlier runs stay in the record with
the parameters they used.

## 1. Why this exists

The pilot results are weak in a way that could mean either of two things — that
7-12B models have little extractable memorization of the canon, or that we are
asking them badly. §2 of the preregistration says the adaptation "changes
prompts/backends only, never test logic or success criteria", and we honoured
that. What we did not check, until now, is whether our *parameters* matched the
ones Bordt et al. used on the models most like ours.

They do not. The authors' own code for the open-model experiments of Table 3 is
in the repository we vendored (`colm-2024-paper-code/notebooks/memorization-tests.ipynb`),
and it sets parameters we left at their library defaults:

```python
# their open-model loop
tabmemcheck.header_test(csv_file, llm, split_rows=split_rows, completion_length=350)
tabmemcheck.row_completion_test(csv_file, llm, num_prefix_rows=8, few_shot=5)

models = [("allenai/OLMo-7B", False),                             # chat_mode=False
          ("google/gemma-2-27b-it", True),
          ("meta-llama/Meta-Llama-3-70B", False),                 # chat_mode=False
          ("Qwen/Qwen1.5-72B", False),                            # chat_mode=False
          ("meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo", True)]
```

| parameter | ours so far | theirs, for open models | consequence |
|---|---|---|---|
| `few_shot` (row completion) | 7 (library default) | **5** | our prompt carries two extra blocks from other datasets |
| `num_prefix_rows` (row completion) | 10 (library default) | **8** | our prompt carries two extra rows |
| `completion_length` (header) | 500 (library default) | **350** | our few-shot blocks are half again as long |
| `config.max_tokens` | 300 header / 100 row / 60 feature | 1000 (library default; not overridden) | ours truncates the answer, and the header test's own statistic is "how many rows were reproduced" |
| `chat_mode` | `True` for every model | **`False` for three of the five** | a different probe altogether |
| `num_queries` | 25-50 | 25 in that loop; Tables 5/6 report 136-250 | power |

The `max_tokens` caps were ours and had a reason — they were chosen to make a
$5 OpenAI budget stretch (`src/run_repro.py`: "a tight limit buys budget
headroom"). That reason does not exist on open weights, and it was carried over
by inertia. On the header test it is not merely conservative: the verdict counts
complete rows reproduced, so capping the answer caps the statistic.

The combined effect of the first three is that our prompts ran to 3.7-4.4k tokens
where the reference protocol's are far shorter. That is also, directly, what made
five cells die of CUDA OOM.

## 2. The correction

For every run from this date, on open-weight models, the reference parameters:

- `few_shot = 5`, `num_prefix_rows = 8` (row completion and first token);
- `completion_length = 350` (header test);
- `max_tokens` at the library default rather than our budget caps.

This is a correction *toward* the published protocol, not a tuning of our own.
It was found by reading the authors' code, it is dated, and the runs made under
the old parameters keep their numbers and their labels. Anyone re-running either
configuration gets what the record says they should.

## 3. The prompting-mode probe (new, preregistered here)

`chat_mode` is not a detail. In chat mode `tabmemcheck` sends a system prompt plus
five to seven few-shot blocks drawn from *other* datasets and then the prefix
rows. In completion mode it sends the prefix rows as raw text, with no system
prompt, no few-shot blocks, and `max_tokens` set to exactly one row's length. The
second is a far more direct question — *continue this text* — and it is what the
authors used for OLMo-7B, Llama-3-70B and Qwen1.5-72B.

We have been asking every model the first way. **This amendment adds the second
as a preregistered comparison**, run on the same models, the same datasets and
the same seed, so that "how much does this model reproduce" is separated from
"how well does this model follow an instruction to reproduce".

Two things follow if completion mode extracts more, and both matter:

- the floor reported in `RESULTS_12B_DIAGNOSTIC.md` would be partly a property of
  our prompting rather than of the models, and that has to be said plainly;
- **the chat-template confound of `AMENDMENT_3` §3 disappears in completion mode.**
  There is no chat template and no system prompt, so Mistral-Nemo and Vikhr-Nemo
  are asked the identical question in the identical form. If H1b's primary
  comparison runs in completion mode, the confound is designed out rather than
  controlled for.

Reported either way. A finding that these models reproduce far more under direct
continuation than under instruction is itself worth stating: it means
instruction-tuned models can hold data they will not hand over when asked
politely, which is the "memorized but not extractable" ambiguity Bordt et al.
flag, made concrete.

## 4. How many queries — the power analysis (`src/power_h1b.py`)

Required by §6 for H3 and equally necessary here. The pilot put Mistral-Nemo at
4/50 and Vikhr-Nemo at 7/50 on iris row completion — 0.08 against 0.14.

| true difference | Cohen's h | queries per arm for 80% power |
|---|---|---|
| 0.08 → 0.14 | 0.193 | 210 |
| 0.08 → 0.16 | 0.250 | 127 |
| 0.08 → 0.20 | 0.354 | 63 |
| 0.26 → 0.50 | 0.501 | 32 |

Row completion cannot ask more questions than the file has rows. With eight
prefix rows the ceilings are: **iris 142**, wine 170, diabetes 760, titanic 883,
california 20 632, adult 32 553. Bordt et al. hit the same wall and report 136 for
iris where they report 250 elsewhere.

So on iris — the only dataset where anything extracts at all — exhausting the
dataset gives **64% power** for a difference the size of the pilot's, and 99% for
a difference of 0.08 → 0.20.

The conclusion is not "add seeds". Seeds do not raise the ceiling on a
dataset-bounded test, and the preregistered three seeds are for few-shot example
selection, not for manufacturing queries. A decisive H1b needs either a larger
true difference or more datasets off the floor, and the prompting-mode probe is
the one intervention that could deliver either.

## 5. Query counts for the next runs

Dataset-bounded, matching the paper's own practice:

| dataset | row completion | first token |
|---|---|---|
| iris | 142 (exhausts the file) | 142 |
| uci-wine | 170 (exhausts the file) | — |
| openml-diabetes | 250 | 250 |
| titanic-train | 250 | — |
| adult-train | 250 | 250 |
| california-housing | 250 | — |

Header: four splits per dataset, as before — it is four queries and its cost is
negligible.

Feature completion stays at 250 on the five datasets that have a designated
feature, and is the cell most likely to stay at zero; it is run because a zero
that was measured is worth something and a zero that was assumed is not.

## 6. What this run decides

Stated before it is run:

- **If completion mode lifts row-completion rates materially above chat mode**
  (any dataset moving from ≤0.05 to ≥0.20), it becomes the primary probe for H1
  and H1b, the confound of `AMENDMENT_3` §3 is designed out, and block B proceeds
  at the query counts in §5.
- **If completion mode changes little**, then the floor is a property of these
  models and not of our prompting. H1's contribution becomes the scale boundary
  reported in `RESULTS_12B_DIAGNOSTIC.md` §2, stated on a protocol that now
  matches the reference in every parameter we could check, with a power analysis
  saying what the null does and does not exclude.

Neither outcome is a failure of the study, and the second is the one the current
evidence points at. Saying so before the run is the point of writing it here.
