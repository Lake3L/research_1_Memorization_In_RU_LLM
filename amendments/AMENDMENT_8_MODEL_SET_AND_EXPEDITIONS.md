# Amendment 8 to PREREGISTRATION.md — models trained from scratch on Russian, a published-verdict control, and three expeditions

**Date:** 2026-09-09. **Scope:** §3 (models) and §9 (compute). Adds two
Russian-centric models trained from scratch and one open-corpus control,
states the hypothesis they test before they are run, and declares three
bounded side studies whose outcome is reported whichever way it falls. No
hypothesis, threshold, test, correction or dataset changes.

---

## 1. Why the model set was incomplete

§3 chose Russian-centric models by base↔adapted pairs, because H1b is a
paired contrast, and listed GigaChat and YandexGPT as API-only exploratory
options, because an API gives no greedy-decoding guarantee and can change
version under a study (§2). Both reasons still hold for the APIs. Both
models have since been released with open weights, which removes the
objection, and session C1 gave a reason to want them.

All three confirmatory Russian-centric models are adaptations: T-lite and
ruadapt continue Qwen2.5-7B, Vikhr-Nemo continues Mistral-Nemo-12B. What an
adaptation can memorize of a Russian file is bounded by what its adaptation
corpus contained, and those corpora are instruction- and continued-pretraining
scale. The base models were trained on Western web crawls where a single
GitHub CSV occurs once. C1 found no Russian dataset memorized by either
member of the Nemo pair, and iris memorized by both. That pattern is what
exposure predicts if the Russian files are absent from both corpora; it says
nothing yet about a model whose pretraining is Russian web at scale. The
model cards of YandexGPT-5-Lite report a from-scratch pretraining of 15
trillion tokens, 60% web pages, Russian and English; GigaChat-20B-A3B is a
from-scratch mixture-of-experts family. If any open model has seen
data.mos.ru exports or a Russian GitHub registry, it is one of these.

## 2. Models added

| model | revision | role | licence |
|---|---|---|---|
| `yandex/YandexGPT-5-Lite-8B-pretrain` | f4aec3abf522c57354f0000e3e30719b3e65bda3 | Russian-centric, from scratch; pretrain member of a pretrain↔instruct pair | YandexGPT-5-Lite-8B agreement (research and commercial use permitted; attribution on redistribution; Russian law) |
| `yandex/YandexGPT-5-Lite-8B-instruct` | b556811768376b46c69caab60c4d1b69df9faaa1 | Russian-centric, from scratch; instruct member | same |
| `ai-sage/GigaChat-20B-A3B-base` | 78724201ba8db03d9fb5f4d54d39df7c2563c9ff | Russian-centric, from scratch, MoE 21B total; base member | MIT |
| `ai-sage/GigaChat-20B-A3B-instruct` | d6a97fca83b088cb636756a0485242e05198998e | instruct member | MIT |
| `allenai/OLMo-7B-hf` | 11fb3186a2e4f681edea621fa8b4345147a9db6a | method control: the one model of our size class with published per-dataset verdicts (Bordt et al. Table 3) and a public, searchable pretraining corpus (Dolma v1.7) | Apache-2.0 |

Revisions are the `sha` of the Hugging Face API on 2026-09-09 and go into
`models.lock`. None of the five is gated.

**Where they enter.** YandexGPT and GigaChat enter H1 and H2 as
Russian-centric models. They do not enter H1b: "llama architecture" on the
YandexGPT card describes the network's shape, not its weights, which were
trained without third-party checkpoints, so `Llama-3.1-8B` is not their
base. Their pretrain↔instruct contrast is a secondary, exploratory analysis
of what instruction tuning does to extraction, the question `AMENDMENT_4`
opened on the prompting side; it is reported, not tested. OLMo enters no
hypothesis: it is run on the canon under `probe` and `h1b_rest` and compared
cell by cell with the published verdicts, and its corpus is queried for the
rows of every frozen dataset (§4, E2). H2's strong form ("negative for every
multilingual control") is evaluated over the controls of §3; YandexGPT and
GigaChat are Russian-centric, not controls, and do not enter it.

**Compute.** YandexGPT-8B loads in 4-bit at about 5 GB. GigaChat-20B-A3B in
nf4 is estimated at 12–13 GB against 15 GB usable on a T4 (21B parameters,
128k vocabulary with untied head; embeddings and head stay in fp16); it runs
alone on its accelerator, after the 1 GB preflight of
`src/check_quantization.py` and a full-size smoke test, and its custom
modelling code is a compatibility risk under transformers 5 that the smoke
test settles before any plan is priced. OLMo-7B-hf loads natively; its 2,048
token context bounds it to the canon.

## 3. The hypothesis, stated before any of them runs

H2's null on the adaptations is explained by exposure, not by adaptation.
Prediction: (i) YandexGPT and GigaChat reproduce iris like every other
model; (ii) if either is positive on a Russian file under `AMENDMENT_7`, the
file's duplication count (E1) is higher than the others'; (iii) if both are
at zero on every Russian file and E1 finds the files essentially unreplicated
on the web, H2 is refuted with the mechanism named, and the block C result is
reported as "not memorized by any model, consistent with single-copy
exposure" rather than as an unexplained null. Outcome (ii) would be the
first positive H2 cell; outcome (iii) is the negative result §10 commits to
publishing. Neither is preferred.

## 4. Three bounded expeditions

Each has a stated cost, a stated deliverable and no influence on the
decision rules; each is reported whichever way it comes out.

**E1 — the duplication covariate.** Kandpal et al. (arXiv:2202.06539) report a
superlinear relation between how often a sequence occurs in training data
and how often it is regenerated, and that unduplicated sequences are "very
rarely regenerated"; Bordt et al. attribute header memorization to the canon's
first rows being "frequently in Jupyter notebooks". For every frozen dataset
we count public copies: exact-string hits for two distinctive rows in GitHub
code search and in the web, and, where the infini-gram index of Dolma v1.7
answers, the count of the same strings in an actual pretraining corpus. The
count is reported beside digits-per-row and the near-duplicate share as the
third entropy covariate, and beside every zero. Cost: one day.

**E2 — corpus composition from primary sources.** For each model in §3 and
§2, what its technical report or model card says about pretraining and
adaptation data: sources, token counts, languages, code and GitHub, open
data, deduplication, cutoff. Deliverable: a table in the block C results
document with quotations and URLs, and the exposure classification
(from-scratch Russian web / adapted / Western) each model gets. For OLMo,
where the corpus is public, the classification is verified by lookup rather
than inferred. Cost: half a day of reading, already delegated and to be
verified.

**E3 — the fresh registry twin** (`AMENDMENT_5` §4, `AMENDMENT_7` R5). A
Russian open-data export of the same construction as the data.mos.ru files
or the govdomains registry — sequential ids, templated consecutive rows,
Cyrillic values — whose records carry a creation date after 2026-06-01, so
that no model with an earlier cutoff can have seen its content. Registered
with source, schema, collection date and hash before any model runs on it;
the pre-cutoff file it mirrors is named at the same time. Cost: search
delegated; collection and registration one session.

## 5. Session order

1. C2: `ru_probe_long` on the Nemo pair (russian_retail, iris anchor).
2. YandexGPT pretrain and instruct, one per accelerator: `probe` and
   `ru_probe` under `AMENDMENT_7` scoring.
3. OLMo-7B-hf on the canon: `probe` and `h1b_rest`.
4. GigaChat base and instruct, alone on a card each, after the smoke test.
5. The Qwen pairs and Llama-3.1-8B as already planned for block B.
6. The fresh twin, on every model that ran on the file it mirrors.

Plans are priced with `src/price_plan.py` at the conservative figure, which
C1 showed to be the right one.

## 6. What does not change

Hypotheses and their thresholds, the four tests, the datasets, the
serialisation rules, the reporting rules, and the base↔adapted pairs of §3,
which remain the confirmatory design of H1b. The additions widen H1 and H2
and add a method check; they narrow nothing.
