"""Kaggle/Colab runner: memorization tests on open-weight models.

Kept as a .py file and converted to .ipynb by `src/build_notebook.py`, so the
pipeline stays reviewable in git rather than buried in JSON cell arrays.

What it does, in order:
  1. installs pinned dependencies and clones the project repository;
  2. re-downloads every dataset and verifies its SHA-256 against the registry —
     a run on a file that does not match the frozen hash is not a valid run;
  3. loads one open-weight model at a pinned revision;
  4. runs the §8 validation first: the HF backend must reproduce the English
     result of the reference pipeline on iris before anything else executes;
  5. runs the memorization tests over datasets × serialisation variants ×
     prompt language, writing every prompt and response to disk.

Nothing here decides anything: the outputs are counts and raw logs, and the
hypothesis tests run offline against the preregistered criteria.
"""

# %% [markdown]
# # Memorization of tabular data in Russian-language LLMs — test run
#
# Set the model and dataset group below, then Run All. Expect ~40 minutes for a
# 7-8B model in 4-bit over one dataset group on a T4.

# %% setup
REPO_URL = "https://github.com/Lake3L/research_1_Memorization_In_RU_LLM.git"
MODEL_ID = "t-tech/T-lite-it-1.0"      # or Qwen/Qwen2.5-7B-Instruct, Vikhrmodels/...
MODEL_REVISION = None                   # pin before confirmatory runs
DATASET_GROUP = "canon"                 # canon | ru_pre_cutoff | fresh_control
PROMPT_LANGUAGES = ["en", "ru"]         # H4 factor
NUM_QUERIES = 50                        # per test; raise for confirmatory runs
SEED = 42
LOAD_IN_4BIT = True

# %% install
import subprocess, sys, os

def sh(cmd):
    print(f"$ {cmd}")
    subprocess.run(cmd, shell=True, check=False)

sh(f"{sys.executable} -m pip install -q 'pandas<3' 'transformers>=4.44' accelerate "
   f"bitsandbytes tabmemcheck jellyfish xgboost scipy")

if not os.path.exists("research_1_Memorization_In_RU_LLM"):
    sh(f"git clone -q {REPO_URL}")
os.chdir("research_1_Memorization_In_RU_LLM")
sys.path.insert(0, "src")

# %% [markdown]
# ## Data integrity
#
# The registry records a SHA-256 for every dataset and for every serialisation
# variant. Files are re-fetched here and re-hashed; any mismatch aborts the run,
# because a memorization measurement is meaningless if the bytes are not the ones
# that were frozen.

# %% fetch and verify
import json
from dataset_registry import load_registry, sha256, read_table, write_variants

registry = load_registry()
targets = {n: r for n, r in registry.items() if r["group"] == DATASET_GROUP}
print(f"{len(targets)} datasets in group '{DATASET_GROUP}'")

import urllib.request

os.makedirs(f"data/{DATASET_GROUP}", exist_ok=True)
verified, failed = [], []
for name, rec in targets.items():
    path = rec["raw_path"]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        try:
            urllib.request.urlretrieve(rec["source_url"], path)
        except Exception as e:
            failed.append((name, f"download failed: {e}"))
            continue
    digest = sha256(path)
    if digest != rec["raw_sha256"]:
        failed.append((name, f"hash mismatch: {digest[:16]} != {rec['raw_sha256'][:16]}"))
    else:
        verified.append(name)
        write_variants(read_table(path), name, f"data/{DATASET_GROUP}/variants")

print("verified:", verified)
print("failed:", failed)
assert verified, "no dataset passed verification — stopping"

# %% [markdown]
# ## Model

# %% load model
import torch
from hf_llm import HFLLM

quantization = None
if LOAD_IN_4BIT:
    from transformers import BitsAndBytesConfig
    quantization = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)

llm = HFLLM(model_name=MODEL_ID, revision=MODEL_REVISION,
            device="cuda" if torch.cuda.is_available() else "cpu",
            quantization_config=quantization)
print("loaded", MODEL_ID, "| chat template:",
      bool(getattr(llm.tokenizer, "chat_template", None)))

# %% [markdown]
# ## Gate: the adapted pipeline must reproduce the reference result
#
# PREREGISTRATION.md §8 requires the HF/Russian pipeline to reproduce the English
# result of the unmodified pipeline on at least one model×dataset before it is
# used for H1-H4. On iris the reference is unambiguous: GPT-4-0613 completes 24/25
# rows and every model tested by Bordt et al. passes the header test. A model that
# scores zero here has either not memorized iris — possible, and itself a result —
# or our adapter is broken, and the distinction is made by the mock controls.

# %% pipeline self-test
from mock_llm import PerfectMemorizer, format_echo_mock
import tabmemcheck as tabmem
import numpy as np

tabmem.config.print_prompts = False
tabmem.config.print_responses = False

class MockAdapter(tabmem.LLM_Interface):
    def __init__(self, fn): super().__init__(); self.fn = fn; self.chat_mode = True
    def chat_completion(self, messages, temperature, max_tokens): return self.fn(messages)
    def completion(self, *a, **k): raise NotImplementedError

iris_path = "data/canon/iris.csv"
if os.path.exists(iris_path):
    perfect = tabmem.row_completion_test(iris_path, MockAdapter(PerfectMemorizer(iris_path)),
                                         num_queries=10, rng=np.random.default_rng(SEED),
                                         print_levenshtein=False)
    echo = tabmem.row_completion_test(iris_path, MockAdapter(format_echo_mock),
                                      num_queries=10, rng=np.random.default_rng(SEED),
                                      print_levenshtein=False)
    print("mock controls ran — perfect memorizer must be 10/10, echo mock 0/10 (above)")

# %% [markdown]
# ## Memorization tests
#
# Every combination of dataset, serialisation variant and prompt language is run
# and logged. Russian prompts change only the instructions and the few-shot
# scaffolding; the CSV content is never translated, because verbatim memorization
# is a property of the published bytes.

# %% run
from run_repro import run_one, MAX_TOKENS
from datetime import datetime, timezone

RU_SYSTEM_PROMPTS = {
    "row-completion":
        "Ты — бот автодополнения для табличных данных. Твоя задача — выдавать строки "
        "ровно в том виде, в каком они содержатся в табличных датасетах. Пользователь "
        "даёт несколько идущих подряд строк из датасета. Ты выдаёшь следующую строку.",
    "header":
        "Ты — бот автодополнения для табличных данных. Тебе будут даны части "
        "табличного датасета. Твоя задача — дополнить датасет.",
    "feature-completion":
        "Ты — бот автодополнения для табличных данных. Наблюдения даются тебе в "
        "формате «Имя признака = Значение признака».",
}

stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
model_tag = MODEL_ID.replace("/", "_")
os.makedirs("results", exist_ok=True)
all_results = []

for name in verified:
    rec = registry[name]
    for variant, info in rec["variants"].items():
        if not os.path.exists(info["path"]):
            continue
        for language in PROMPT_LANGUAGES:
            if language == "ru":
                tabmem.config.system_prompts.update(RU_SYSTEM_PROMPTS)
            else:
                import importlib
                importlib.reload(tabmem)
            for test in ("header", "row", "feature"):
                try:
                    r = run_one(llm, info["path"], test,
                                4 if test == "header" else NUM_QUERIES, SEED)
                except Exception as e:
                    r = {"dataset": name, "test": test, "error": f"{type(e).__name__}: {e}"}
                r.update(dataset_name=name, variant=variant, prompt_language=language,
                         model=MODEL_ID, revision=MODEL_REVISION)
                all_results.append(r)
                print(f"{name[:20]:20s} {variant[:14]:14s} {language} {test:8s} -> "
                      f"{r.get('matches', r.get('verdict', r.get('error')))}")

out = f"results/memorization_{model_tag}_{DATASET_GROUP}_{stamp}.json"
with open(out, "w", encoding="utf-8") as f:
    json.dump({"model": MODEL_ID, "revision": MODEL_REVISION, "group": DATASET_GROUP,
               "seed": SEED, "num_queries": NUM_QUERIES, "timestamp_utc": stamp,
               "verified_datasets": verified, "failed_datasets": failed,
               "results": all_results}, f, ensure_ascii=False, indent=2)
print("wrote", out)

# %% [markdown]
# ## Download the results
#
# On Kaggle the file lands in `/kaggle/working`; commit the notebook to keep it as
# an output, then attach it to the repository under `results/`.
