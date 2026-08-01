# Memorization of Tabular Data in Russian-Language LLMs

> Draft grown incrementally after each experiment (rule 1.5 of the working plan).
> Target structure per plan §2.8. Notes in brackets are TODO markers, not prose.

## Abstract

[После первых результатов.]

## 1. Introduction

[Проблема: контаминация табличных бенчмарков установлена для англоязычных моделей (Bordt et al., COLM 2024) и активно развивается (Silvestri+ 2025; Ronval+ IDA 2025; Gorla & Puduppully ICML 2026), но вся литература англоцентрична. Русскоязычные LLM оцениваются на табличных задачах без какого-либо аудита контаминации. Пробел ×2: русские модели не проверялись, русские датасеты не проверялись, и сама методика детекции не проверялась на чувствительность к языку промпта — при том, что «язык маскирует контаминацию» уже показано для текстовых бенчмарков (Yao+ EMNLP 2024; Abbas+ 2026, арабский).]

## 2. Related Work

[Три блока: (1) меморизация/контаминация табличных данных — Bordt 2403.06644+2404.06209, Silvestri 2510.20351, Ronval IDA 2025, Gorla 2602.04031, ICLMEM 2606.31208, LLMTabBench 2605.24417, Ward 2512.08875 (privacy-угол, отделить), сенсорный перенос 2406.05900; (2) кросс-язычная контаминация — Yao 2406.13236, Abbas 2601.14994, multilingual report 2410.16186, CoDeC (baseline); (3) русскоязычная оценка LLM — MERA 2401.04531 (профилактика, не аудит), техотчёты моделей. Surveys: 2502.14425, 2406.14644.]

## 3. Methods

[Тесты Bordt (header/row/feature completion, first token) + адаптация: HF-модели, русские промпты. Конфаунд «дословная меморизация vs знание статистик»: таксономия уровней знания Bordt + приём real/like/swapped/obfuscated из Silvestri+. Датасеты: западный канон / русские открытые / свежий контроль (дата сбора зафиксирована). Языковой фактор: рус/англ промпты на одной модели.]

### 3.x Choice of dataset formats for the few-shot contrast

Bordt et al. present four formats. They are not interchangeable instruments, and the
difference decides how H3 must be measured.

- *original* — the published file, byte for byte.
- *perturbed* — same column names, same units, numeric values nudged by one unit in the
  last digit (iris `5.1, 3.5` → `5.2, 3.4`). Semantics untouched; only the exact strings
  change.
- *task* — columns renamed to natural language (`sepal_length` → `Length of Sepal (cm)`),
  categories recoded (`Iris-setosa` → `Setosa`), values rounded, and the dataset name
  removed from the system prompt. Domain knowledge remains fully usable; identification
  of the specific dataset is what breaks.
- *statistical* — features standardised, multiplied by −3.33, noised, and renamed to
  `X1…Xn` with the target as `Y` (iris row 1 becomes `2.65, −2.76, 4.69, 4.33, 0`).
  Nothing identifies the domain: petal length is no longer recognisable as a petal.

Perturbed and task hold the classification problem *and its semantics* fixed while
breaking verbatim recall, so an accuracy drop there is attributable to memorization.
The statistical format additionally removes every semantic prior an LLM could bring,
which is a different manipulation: it changes what the model can do, not just what it
can recall.

Recomputing the authors' own released responses (see `RESULTS_TABLE4.md`) shows this is
not a theoretical concern. Averaged over both models and all datasets, the statistical
format costs **8.7 accuracy points on datasets published after the training cutoff** —
data no model could have memorized — against **+0.2 points for perturbed and task** on
the same datasets. A memorization estimate built on the statistical format therefore
inherits a large domain-knowledge penalty that has nothing to do with contamination.

Accordingly our H3 contrast is *original vs perturbed/task*, preregistered before this
analysis, and *statistical* is reported separately as a difficulty control: it is expected
to cost a similar amount on seen and fresh data, and a divergence there would itself be a
finding. Note that the paper's own control (its Tables 9-10, logistic regression and
gradient boosting scoring equally on all four formats) establishes that the *statistical
task* is preserved — it cannot establish that the *LLM's* task is preserved, because the
baselines it uses never read feature names in the first place.

## 4. Experiments

[Модели с HF-ревизиями, 3 seed'а, температуры, форматы сериализации.]

## 5. Results

[По гипотезам H1–H4; поправка Holm/BH на множественные сравнения; отрицательные результаты публикуются.]

## 6. Limitations

[API-модели без logprobs (если включим); размер выборки русских датасетов; остаточная неопределённость конфаунда; переносимость на другие нелатинские языки. Источник — LOG.md.]

## 7. Conclusion

## Reproducibility Statement

[Одна команда из чистого клона; seeds; HF commit hashes; requirements с точными версиями; даты сбора данных.]
