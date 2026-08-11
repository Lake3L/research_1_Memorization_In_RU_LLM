# Amendment 2 to PREREGISTRATION.md — the canon is re-frozen on its published bytes

**Date:** 2026-08-11. **Scope:** the SHA-256 of five Western-canon files. Nothing
else: no dataset is added or removed, no Russian dataset is touched, no derived
serialisation variant changes, and no hypothesis, criterion or decision rule is
affected.

## What happened

`git config core.autocrlf` is `true` on the machine where the datasets were
prepared, which is the Windows default. When `interpretml/LLM-Tabular-Memorization-Checker`
was cloned there in order to install `tabmemcheck`, git rewrote every `\n` in its
data files to `\r\n` on checkout. The canon files were frozen from that clone on
2026-08-10. Five of the six therefore carry Windows line endings that the
published files do not have; `titanic-train` was downloaded over HTTP and escaped.

The freeze was caught by its own hash check. The block A notebook run on Kaggle
on 2026-08-11 stopped at the data step with five hash mismatches and refused to
proceed — which is what the check is for.

| dataset | frozen 2026-08-10 (CRLF) | published bytes (LF) | md5 recorded in Amendment 1 | true md5 |
|---|---|---|---|---|
| `iris` | `9194e2b71f7144e7…` | `abd4f9391ef31795…` | `bab8c78b` | `9f44d5c5` |
| `adult-train` | `c18b1aa5bb34303b…` | `b40dc2181c59d792…` | `82247b7d` | `12a09c3e` |
| `california-housing` | `b214a65099c1dcc2…` | `8a3727f4cf54ac1a…` | `e2727d25` | `d1c47305` |
| `openml-diabetes` | `698c203a14aa3194…` | `b78029447fae2743…` | `f2906818` | `b43dd020` |
| `uci-wine` | `6cb704889e69841f…` | `21a3bb1b675594a2…` | `de2633a7` | `bf10dd49` |
| `titanic-train` | `4a437fde05fe5264…` | unchanged | — | — |

The md5 values in Amendment 1 were offered as evidence that our files were
"byte-identical to the copy shipped with tabmemcheck and with the COLM paper
code". They were the md5 values of a Windows checkout of that copy. The claim was
therefore false as stated, and it is corrected here rather than quietly rewritten:
the entries in Amendment 1 now cite the bytes served by tabmemcheck commit
`7dbeaac5` and by the tabmemcheck 0.1.6 wheel on PyPI, which were verified to
agree with each other byte for byte on 2026-08-11.

## What this does and does not invalidate

**No measurement is invalidated, and this was checked rather than assumed.**
`tabmemcheck` reads CSVs through Python's text mode, whose universal-newline
handling collapses `\r\n` to `\n` at load. Feeding it the CRLF and the LF form of
iris produces identical output from `load_csv_string` and `load_csv_rows`, so the
prompts sent to a model and the strings compared against its answers are the same
either way. The reproduction of Bordt et al. recorded in `RESULTS_GATE.md` §2
therefore stands unchanged, as do the mock controls and the Table 4 re-analysis.

What was broken is the bookkeeping, and it was broken in a way that matters:

- the hashes in the frozen document described files that exist only on one
  Windows machine, so nobody else could have reproduced the freeze;
- the pipeline could not verify its own data on any non-Windows machine, which is
  exactly where every confirmatory run has to happen (no local GPU);
- the provenance claim about byte-identity with the reference implementation was
  wrong, and provenance is the whole basis for saying a file predates a model.

Had the data step been made lenient instead of fixed, the run would have
proceeded on files whose published form we could no longer point to. The reason
to write this down is that the failure was invisible from inside the machine that
caused it: every check passed locally, for eight days.

## What changed in the code

1. `src/fetch_data.py` verifies the frozen hash **as part of choosing a source**,
   not after one has been chosen. Previously the first source that did not raise
   was accepted, so the local `tabmemcheck` copy — right rows, wrong bytes — won
   and the correct source was never tried. Sources now fall through on a hash
   mismatch, and the canon's primary source is the pinned GitHub commit, with the
   installed package demoted to an offline fallback.
2. `.gitattributes` marks `*.csv` and `*.jsonl` as binary, so our own repository
   cannot be converted on checkout. This cannot protect a clone of somebody
   else's repository, which is where the damage came from; the authority on
   dataset bytes is `src/fetch_data.py` and the registry, never a checkout.
3. `iris` was being registered by hand rather than by a committed script — the
   only dataset of which that was true, and the reason it survived the first
   re-freeze attempt unchanged. It is now in `src/prepare_canon.py` with the
   others.

## Reproduction

```
python src/fetch_data.py --group canon --refreeze   # take the published bytes
python src/prepare_canon.py                          # re-register with new hashes
python src/make_amendment.py                         # regenerate Amendment 1
python src/dataset_registry.py                       # integrity: OK
python src/run_hf_gate.py --mock perfect             # 115/115, 6/6 header passes
python src/run_hf_gate.py --mock echo                # zero on every verbatim cell
```
