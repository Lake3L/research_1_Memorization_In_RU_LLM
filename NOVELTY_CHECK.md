# Протокол проверки новизны — Проект 1

**Тема:** Меморизация табличных данных в русскоязычных LLM.
**Дата прохождения:** 2026-07-13 (первый проход; повторный — за неделю до подачи препринта, см. раздел 1.1 и чек-лист research-plan.md).
**Статус:** ЗАВЕРШЁН, все 5 шагов. **Вердикт: ниша свободна, работаем** (условия позиционирования — в конце документа).

---

## 0. Проверка первоисточников (⚠️-пункты плана)

Все три статьи существуют, ID и названия в плане точные (проверено через arXiv API, 2026-07-13):

| arXiv ID | Название | Дата | Venue |
|---|---|---|---|
| 2403.06644 | Elephants Never Forget: Testing Language Models for Memorization of Tabular Data | 2024-03-11 | TRL Workshop @ NeurIPS 2023 |
| 2404.06209 | Elephants Never Forget: Memorization and Learning of Tabular Data in LLMs | 2024-04-09 | COLM 2024 |
| 2512.08875 | When Tables Leak: Attacking String Memorization in LLM-Based Tabular Data Generation | 2025-12-09 | PoPETs |

Уточнение по Ward et al.: полное название — «...Attacking String Memorization in LLM-Based Tabular Data Generation», опубликована в Proceedings on Privacy Enhancing Technologies. Это privacy-угол (атаки на меморизацию при генерации синтетики), не contamination-угол — соседняя, не конкурирующая ниша.

Ссылка на инструмент tabmemcheck — проверяется на шаге 5 (GitHub).

## Шаг 2. Цитирование вперёд (Semantic Scholar API)

Проверены **все** цитирующие работы трёх центральных статей (выгрузка 2026-07-13):
- arXiv:2404.06209 (COLM-версия): 35 цитирований;
- arXiv:2403.06644 (workshop-версия): ~30 цитирований (пересечение с первым списком, суммарно 58 уникальных);
- arXiv:2512.08875 (Ward): 2 цитирования.

**Ни одной работы про меморизацию/контаминацию табличных данных на неанглийском языке или в русскоязычных моделях не найдено.** Ближайшие соседи (заслуживают цитирования, но не конкурируют):

| Работа | Что делает | Угроза новизне |
|---|---|---|
| arXiv:2606.31208 «Probing Memorization of Tabular In-Context Learning» (2026) | Парметрическая меморизация в табличных foundation-моделях (TabPFN-класс, ICL): probing-фреймворк ICLMEM, membership через контролируемый файнтюнинг. Не про претрейн-контаминацию LLM, не про языки. | Низкая |
| arXiv:2406.05900 «LLMs Memorize Sensor Datasets!» (2024) | Перенос тестов Bordt на сенсорные датасеты (HAR). Прецедент доменного переноса методики — полезный шаблон, англоязычный. | Низкая |
| arXiv:2510.12950 «Memorization Risk in Healthcare Foundation Models» (2025) | Privacy-тесты меморизации EHR-моделей. Другой вопрос (приватность, не контаминация оценки). | Низкая |
| arXiv:2502.14425 «A Survey on Data Contamination for LLMs» (2025); arXiv:2406.14644 (ACL 2024, survey) | Обзоры контаминации — источник для Related Work; мультиязычный пробел в них не закрыт. | Нет |
| arXiv:2606.26021 «Privacy Vulnerabilities of Attention Layers in Tabular Foundation Models» (2026) | Цитирует Ward; privacy-атаки на табличные FM. | Нет |

## Шаг 3. Граф связанных работ

Connected Papers программно недоступен (SPA, API отдаёт HTML-заглушку) — **замена:** Semantic Scholar Recommendations API (та же функция: соседи по графу похожести), 30 работ вокруг arXiv:2404.06209. Релевантные соседи за последние 12 месяцев — всё те же 2606.31208 и общие работы о меморизации LLM; **ни одной русскоязычной/мультиязычной табличной**. Единственные касания мультиязычности: arXiv:2606.03291 (multilingual unlearning — другая задача). При желании граф Connected Papers можно посмотреть глазами в браузере при повторном проходе перед подачей.

## Шаг 1. Ключевые слова (Google Scholar, arXiv full-text, RU-источники)

Выполнено 25 запросами (18 веб-поиск EN+RU, 3 arXiv full-text, 4 Semantic Scholar; полный список — в конце раздела). Проверка ссылок по первоисточникам выполнена.

### Прямо в нише «табличная меморизация/контаминация» (все — только английский)

| Работа | Что делает | Угроза |
|---|---|---|
| Silvestri et al. «Evaluating Latent Knowledge of Public Tabular Datasets in LLMs», arXiv:2510.20351 (2025, рев. 03.2026) | Детекция контаминации через MCQ-пробы по 4 вариантам датасета (real/like/swapped/obfuscated) + McNemar; 7 моделей (Mistral 7B, Qwen 7–32B, Llama 8B/70B, GPT-OSS), 8 датасетов (Adult, Titanic, Iris, Diabetes и др.) + синтетический контроль. **Проверено по полному тексту (2026-07-13): языкового аспекта нет, всё на английском.** Их схема «real vs like vs obfuscated» — готовый приём для нашего конфаунда «статистики vs дословно». | **Средняя** — конкурирующая методология детекции; читать целиком на неделе 1 |
| Gorla, Puduppully «The Illusion of Generalization in Tabular Language Models», arXiv:2602.04031, **ICML 2026** | «Обобщение» Tabula-8B объясняется утечкой train/test; near-zero lift над majority baseline. | **Средняя** — пересекается с тезисом «контаминация искажает few-shot оценку» |
| Ronval, Dupont, Nijssen «Detection of LLM Contamination with Tabular Data», IDA 2025, LNCS 15669 | Алгоритмы детекции с пертурбацией примеров; закрытые/крупные модели контаминированы сильнее. | **Средняя** — расширяет инструментарий Bordt, цитировать |
| Capano, Böhler «Probing Memorization of Tabular ICL» (ICLMEM), arXiv:2606.31208 (06.2026) | См. шаг 2. Табличные FM, не мультиязычные LLM. | Средне-низкая |

### Кросс-язычная контаминация (НЕ табличная — ближайшая угроза для H4 о языке промпта)

| Работа | Что делает | Угроза |
|---|---|---|
| Yao et al. «Data Contamination Can Cross Language Barriers», **EMNLP 2024**, arXiv:2406.13236 | Контаминация переведёнными бенчмарками завышает скор и обходит детекторы. Текстовые бенчмарки, не таблицы, без русского. | **Средняя** — прецедент «язык × контаминация»; H4 позиционировать как перенос на табличный/русский домен |
| Abbas et al. «Obscuring Data Contamination Through Translation: Evidence from Arabic Corpora», arXiv:2601.14994 (2026) | Перевод на арабский подавляет стандартные индикаторы контаминации; Translation-Aware Detection. | **Средне-высокая для H4** — концептуально то же «язык маскирует меморизацию»; обязательный related work и точка дифференциации |
| «Contamination Report for Multilingual Benchmarks», arXiv:2410.16186 (2024) | Почти все модели контаминированы мультиязычными бенчмарками. Не таблицы. | Низко-средняя |
| «Shared Path: … Memorization in Multilingual LLMs through Language Similarities», arXiv:2505.15722 (2025) | Меморизация в 95 языках vs языковая близость; текстовая. | Низко-средняя |

### Русскоязычная экосистема

- **MERA** (arXiv:2401.04531, ACL 2024) и MERA Multi (arXiv:2511.15552): закрытые тест-сеты и watermarking как *профилактика* утечки — пост-хок аудита меморизации нет, таблиц нет. Угроза: нет, обязательный контекст.
- Техотчёты T-pro 2.0 (arXiv:2512.10430), Vikhr (arXiv:2405.13929), GigaChat (arXiv:2506.09440), ruadapt (arXiv:2412.21140): аудита их меморизации третьими лицами **не существует** — это наши объекты исследования.
- **LLMTabBench**, arXiv:2605.24417 (05.2026, команда предположительно российская): few-shot табличная классификация. **Проверено по полному тексту (2026-07-13):** memorization/recognition-пробы в работе ЕСТЬ (recognition test, cell-level reconstruction lift), но модели — GPT-4o-mini, Qwen3, TabPFN-3, TabICL; все 91 датасет англоязычные, русских моделей/данных/промптов нет. Угроза: низкая; обязателен в Related Work.

### Пусто (ключевое для вердикта)

- Расширения/адаптации tabmemcheck на другие языки — не найдено.
- Меморизация/контаминация табличных данных на любом неанглийском языке — не найдено.
- Русскоязычные публикации (Habr, «Диалог», AIST, AIRI) про меморизацию табличных данных — не найдено.
- Аудит контаминации T-lite/T-pro/Vikhr/GigaChat/YandexGPT/ruadapt третьими лицами — не найдено.

<details><summary>Полный список запросов шага 1 (для воспроизводимости протокола)</summary>

Веб-поиск: (1) "memorization" tabular data LLM large language models 2025 contamination; (2) "data contamination" tabular datasets language models few-shot classification benchmark; (3) data contamination memorization Russian language models T-lite Vikhr GigaChat benchmark; (4) multilingual cross-lingual data contamination LLM benchmark evaluation detection 2025; (5) prompt language effect memorization contamination detection LLM English vs non-English; (6) меморизация табличных данных языковые модели контаминация бенчмарк русский; (7) header test row completion feature completion test tabmemcheck LLM tabular memorization; (8) MERA benchmark contamination test leakage Russian LLM evaluation ruadapt memorization; (9) контаминация данных LLM утечка тестовых данных Habr Диалог конференция русскоязычные модели; (10) few-shot tabular classification LLM contamination bias inflated performance memorized datasets; (11) tabmemcheck multilingual OR Russian OR non-English extension tabular memorization test; (12) "When Tables Leak" Ward tabular contamination arXiv cited follow-up; (13) утечка тестовых данных бенчмарк LLM русский язык оценка моделей Диалог AIST статья; (14) arXiv Russian benchmark contamination ruMMLU rulm "contamination" evaluation Russian-language LLM 2025; (15) "Probing Memorization of Tabular In-Context Learning" arXiv; (16) "Detection of Large Language Model Contamination with Tabular Data" Dupont IDA 2025; (17) "cross-lingual memorization" OR "multilingual memorization" large language models training data 2025 2026; (18) запоминание обучающих данных нейросетевые языковые модели русскоязычные исследование меморизация. arXiv full-text: (19) "tabular data" memorization contamination; (20) memorization Russian language model; (21) "Probing Memorization of Tabular In-Context Learning". Semantic Scholar: (22, 25) поиск по фразам — HTTP 429; (23) цитирования 2403.06644; (24) цитирования 2404.06209.

</details>

## Шаг 4. OpenReview (ICLR 2026, NeurIPS 2025, ICML 2026, COLM, ARR)

Рабочий формат API: `https://api2.openreview.net/notes/search?query=<...>&content=all&group=all&source=all` (вариант `?term=` отдаёт `searchUnavailable`). 18 запросов, релевантность проверялась по содержимому нот.

| Работа | Venue | Суть | Угроза |
|---|---|---|---|
| Ronval, Dupont, Nijssen «Detection of LLM Contamination with Tabular Data» | **IDA 2025** (Springer LNCS 15669, forum hM8CtXgRoy) | Контаминация широкого спектра LLM (не только GPT) на таблицах, оси knowledge/memorization + парсер ответов. Крупнее и закрытее модель → вероятнее контаминация. Языка нет. | **Средняя** — закрывает ось «много открытых LLM», не закрывает русские модели/данные/язык промпта |
| Abbas et al. «Obscuring Data Contamination Through Translation» (арабский) | **Сабмишен ICLR 2026** (forum omg9K6lI93) | Перевод бенчмарков на арабский снижает детектируемость контаминации. | **Средняя** — ближайший концептуальный сосед H4, но NLP-корпуса, не таблицы |
| «Probing Memorization of Tabular ICL» (ICLMEM) | FMSD и MemFM @ **ICML 2026** (posters) | Табличные foundation-модели, не LLM. | Низкая |
| CoDeC «Detecting Data Contamination in LLMs via In-Context Learning» | **ICLR 2026 Poster** (forum YlpaaYxx4t) | Общий ICL-метод детекции, не таблично-специфичный. | Низкая — кандидат в baseline для сравнения |
| TrustGen (рус. trustworthiness-бенчмарк), бенчмарк на рус. WildChat-1M | Сабмишены ICLR 2026 | Русская LLM-оценка активна, но контаминации/таблиц не касается. | Нет |

Пустые запросы: «tabmemcheck» (0), follow-up'ы «When Tables Leak» (0), «contamination Russian» / «MERA contamination» (релевантного нет).

## Шаг 5. GitHub

**Оригинальный инструмент Bordt et al. (⚠️-пункт плана проверен):**
- Репозиторий: **https://github.com/interpretml/LLM-Tabular-Memorization-Checker** (владелец interpretml, PyPI-пакет `tabmemcheck`; авторы Bordt, Nori, Caruana).
- 38 звёзд, **5 форков — все ahead_by: 0** (чистые снапшоты, адаптаций нет), 0 открытых issues, последний push **2025-02-10** (инструмент не развивается ~1.5 года — адаптация под HF-модели и русский целиком на нас, как план и предполагал).
- Вся история issues/PR (4 шт., закрыты): поддержка Anthropic API, мелкие фиксы. Упоминаний других языков нет. Dependents: 0 пакетов.
- Реальные пользователи по code search: autoelicit, llm-elicited-priors (берут util-функцию), llm-gwas-causal-genes (contamination-check в GWAS). Все англоязычные.

**Поиск ниши в коде:** «tabmemcheck» repos → 0 (кроме оригинала); «data contamination Russian LLM», «меморизация табличных», «MERA contamination» и др. → 0 релевантных. В репозиториях MERA собственных исследований контаминации нет (только унаследованный n-gram janitor из lm-evaluation-harness). Русскоязычный веб (Habr и пр.) — пусто.

---

## ВЕРДИКТ (2026-07-13): ниша СВОБОДНА — работаем

По критерию плана («бросай только если найдена работа: то же самое, на тех же данных, с теми же выводами») — таких работ нет. Целевая комбинация **«меморизация табличных данных × русскоязычные LLM × русскоязычные датасеты × язык промпта»** не занята ни статьёй, ни кодом, ни сабмишеном текущих циклов.

**Обязательные условия позиционирования** (иначе новизну съедят соседи):

1. **Ось «не-GPT модели» уже закрыта** (Ronval, IDA 2025). Вклад формулировать через русскоязычные модели, русскоязычные данные и языковое измерение — «мы первые прогнали открытые LLM» новизной не является.
2. **H4 (язык промпта) позиционировать как перенос** установленного эффекта «язык маскирует контаминацию» (Yao+, EMNLP 2024; Abbas+, ICLR 2026 sub — арабский) **на табличный домен и русский язык** — не как открытие эффекта с нуля.
3. **Related Work обязан включать волну 2025–2026:** Silvestri+ (2510.20351), Ronval+ (IDA 2025), Gorla & Puduppully (ICML 2026), ICLMEM (2606.31208), CoDeC (ICLR 2026), LLMTabBench (2605.24417), Ward+ (2512.08875), Yao+ (2406.13236), Abbas+ (2601.14994), surveys 2502.14425 / 2406.14644.
4. **Методологические заимствования:** схема real/like/swapped/obfuscated-вариантов (Silvestri+) — для конфаунда «статистики vs дословно»; CoDeC — как baseline-метод сравнения.

**Риск темпа:** 4+ новых работы в нише за 2025–2026, последняя — июнь 2026. Повторный проход протокола перед подачей препринта обязателен (заложен в план, раздел 5).
