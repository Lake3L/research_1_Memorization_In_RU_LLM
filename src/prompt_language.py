"""The EN↔RU manipulation of H4, in one place.

PREREGISTRATION.md §6 (H4) allows exactly one thing to change between the two
arms: the instruction scaffolding. The CSV bytes are never translated, because
verbatim memorization is a property of the published bytes; only the system
prompts that wrap them switch language. Those prompts are therefore an
instrument of the study and belong in reviewable, diffable code — not in a
notebook cell where a reviewer cannot see what was actually sent.

The translations are literal. Where English says "autocomplete bot" the Russian
says "бот автодополнения", not a more idiomatic Russian phrasing: a fluent
rewrite would confound the language factor with a prompt-quality factor, and we
would not be able to tell which one moved the numbers.

Switching back to English restores the values `tabmemcheck` loaded from its own
resources. Reloading the module instead — the obvious-looking alternative —
does not work: it rebinds `tabmemcheck.config`, silently discarding
`max_tokens`, `print_prompts` and every other setting the run depends on, and
leaves already-created objects bound to the old class.
"""

import copy

import tabmemcheck as tabmem

# Snapshot taken at import, before anything can have modified it.
EN_SYSTEM_PROMPTS = copy.deepcopy(tabmem.config.system_prompts)

RU_SYSTEM_PROMPTS = {
    "header":
        "Ты — бот автодополнения для табличных данных. Тебе будут даны части "
        "табличного датасета. Твоя задача — дополнить датасет.",
    "row-completion":
        "Ты — полезный бот автодополнения для табличных данных. Твоя задача — "
        "выдавать строки в том виде, в каком они содержатся в табличных "
        "датасетах. Пользователь даёт несколько идущих подряд строк из "
        "табличного датасета. Ты выдаёшь следующую строку из датасета.",
    "feature-completion":
        "Ты — полезный бот автодополнения для табличных данных. Тебе будет дана "
        "часть наблюдения из датасета для машинного обучения. Твоя задача — "
        "дополнить это наблюдение из датасета. Наблюдения даются тебе в формате "
        "«Имя признака = Значение признака».",
    "generic-csv-format":
        "Ты — полезный бот автодополнения для табличных данных.\n\nТвоя задача — "
        "дополнять строки табличных датасетов. Строки даются тебе в том виде, в "
        "каком они содержатся в csv-файлах разных популярных датасетов.\n",
    "sample":
        "Ты — экспертный ассистент по табличным данным. Ты выдаёшь случайные "
        "примеры из разных датасетов. Пользователь даёт название датасета и "
        "имена признаков. Ты выдаёшь пример из датасета.",
    "cond-sample":
        "Ты — экспертный ассистент по табличным данным. Ты выдаёшь условные "
        "примеры из разных датасетов. Пользователь даёт название датасета, имена "
        "признаков, а также значения некоторых из признаков. Ты выдаёшь значения "
        "остальных признаков.",
    "feature-names":
        "Ты — экспертный ассистент по табличным данным. Твоя задача — "
        "перечислять имена признаков разных датасетов. Пользователь даёт "
        "описание датасета и некоторые имена признаков. Ты выдаёшь имена "
        "остальных признаков.",
    "dataset-name":
        "Ты — экспертный ассистент по табличным данным. Твоя задача — назвать "
        "датасет. Пользователь даёт начальные строки csv-файла, включая имена "
        "признаков. Ты называешь датасет.",
    "predict":
        "Ты — экспертный ассистент по табличным данным. Ты делаешь предсказания "
        "на разных датасетах. Пользователь даёт название датасета, имена "
        "признаков, а также значения всех признаков кроме одного. Ты выдаёшь "
        "предсказание для отсутствующего признака (целевого).",
}

_MISSING = set(EN_SYSTEM_PROMPTS) - set(RU_SYSTEM_PROMPTS)
assert not _MISSING, f"no Russian version of the system prompts: {sorted(_MISSING)}"


def set_language(language: str) -> dict:
    """Switch the instruction scaffolding. Returns the prompts now in force."""
    if language == "en":
        prompts = copy.deepcopy(EN_SYSTEM_PROMPTS)
    elif language == "ru":
        prompts = copy.deepcopy(EN_SYSTEM_PROMPTS)
        prompts.update(RU_SYSTEM_PROMPTS)
    else:
        raise ValueError(f"unknown prompt language: {language}")
    tabmem.config.system_prompts = prompts
    return prompts


if __name__ == "__main__":
    for language in ("en", "ru"):
        prompts = set_language(language)
        print(f"[{language}] row-completion: {prompts['row-completion'][:80]}...")
    assert set_language("en") == EN_SYSTEM_PROMPTS, "switching back is not lossless"
    print("\nlanguage switch is reversible")
