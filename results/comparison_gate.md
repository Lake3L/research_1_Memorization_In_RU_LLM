
### Requested `gpt-4-0613` → served `gpt-4-0613` (seed 42, 63 calls) — reproduction

| test | dataset | ours | rate [95% CI] | paper | paper rate | paper in CI |
|---|---|---|---|---|---|---|
| row | iris.csv | 24/25 | 0.96 [0.80, 1.00] | 125/136 | 0.92 | yes |
| feature | openml-diabetes.csv | 22/25 | 0.88 [0.69, 0.97] | 243/250 | 0.97 | yes |
| feature | adult-train.csv | ERROR | — | — | — | — |

### Requested `gpt-3.5-turbo-16k` → served `gpt-3.5-turbo-0125` (seed 42, 370 calls) — reproduction

| test | dataset | ours | rate [95% CI] | paper | paper rate | paper in CI |
|---|---|---|---|---|---|---|
| header | iris.csv | pass (10 rows) | — | pass | — | yes |
| header | uci-wine.csv | pass (5 rows) | — | pass | — | yes |
| header | openml-diabetes.csv | pass (5 rows) | — | pass | — | yes |
| header | adult-train.csv | pass (3 rows) | — | pass | — | yes |
| header | california-housing.csv | pass (4 rows) | — | pass | — | yes |
| feature | openml-diabetes.csv | 49/50 | 0.98 [0.89, 1.00] | 237/250 | 0.95 | yes |
| feature | uci-wine.csv | 18/50 | 0.36 [0.23, 0.51] | 77/178 | 0.43 | yes |
| feature | adult-train.csv | 0/50 | 0.00 [0.00, 0.07] | 0/250 | 0.00 | yes |
| feature | california-housing.csv | 0/50 | 0.00 [0.00, 0.07] | 0/250 | 0.00 | yes |
| row | iris.csv | 20/50 | 0.40 [0.26, 0.55] | 35/136 | 0.26 | NO |
| row | openml-diabetes.csv | 2/25 | 0.08 [0.01, 0.26] | 18/250 | 0.07 | yes |
| row | adult-train.csv | 0/25 | 0.00 [0.00, 0.14] | 0/250 | 0.00 | yes |
| first_token | iris.csv | 30/50 | 0.60 [0.45, 0.74] | 88/136 | 0.65 | yes |

### Requested `gpt-4o-mini` → served `gpt-4o-mini-2024-07-18` (seed 42, 14 calls) — extension — no published column for this model, not counted below

| test | dataset | ours | rate [95% CI] | paper | paper rate | paper in CI |
|---|---|---|---|---|---|---|
| header | iris.csv | pass | — | — | — | n/a |
| header | uci-wine.csv | fail | — | — | — | n/a |
| header | openml-diabetes.csv | pass | — | — | — | n/a |
| header | adult-train.csv | ERROR | — | — | — | — |

### Requested `gpt-4o-mini` → served `gpt-4o-mini-2024-07-18` (seed 42, 370 calls) — extension — no published column for this model, not counted below

| test | dataset | ours | rate [95% CI] | paper | paper rate | paper in CI |
|---|---|---|---|---|---|---|
| header | iris.csv | pass (7 rows) | — | — | — | n/a |
| header | uci-wine.csv | fail (0 rows) | — | — | — | n/a |
| header | openml-diabetes.csv | pass (6 rows) | — | — | — | n/a |
| header | adult-train.csv | pass (1 rows) | — | — | — | n/a |
| header | california-housing.csv | fail (0 rows) | — | — | — | n/a |
| feature | openml-diabetes.csv | 1/50 | 0.02 [0.00, 0.11] | — | — | — |
| feature | uci-wine.csv | 1/50 | 0.02 [0.00, 0.11] | — | — | — |
| feature | adult-train.csv | 0/50 | 0.00 [0.00, 0.07] | — | — | — |
| feature | california-housing.csv | 0/50 | 0.00 [0.00, 0.07] | — | — | — |
| row | iris.csv | 13/50 | 0.26 [0.15, 0.40] | — | — | — |
| row | openml-diabetes.csv | 0/25 | 0.00 [0.00, 0.14] | — | — | — |
| row | adult-train.csv | 0/25 | 0.00 [0.00, 0.14] | — | — | — |
| first_token | iris.csv | 31/50 | 0.62 [0.47, 0.75] | — | — | — |

**Cells consistent with the paper: 14/15**
