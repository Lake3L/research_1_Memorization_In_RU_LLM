"""Kaggle/Colab runner for block A: does the adapted pipeline reproduce the
English result of the unmodified one (PREREGISTRATION.md §8, second half).

Kept as a .py file and converted to .ipynb by `src/build_notebook.py`, so the
pipeline stays reviewable in git rather than buried in JSON cell arrays.

The notebook is deliberately thin. Everything that decides anything lives in
`src/`, is unit-checked, and has already been run end to end against both mocks
on a laptop with no GPU (`python src/run_hf_gate.py --mock perfect|echo`). What
is left here is what genuinely needs a GPU: install, fetch, load the weights,
run, download the outputs. A notebook cell that nobody can diff is a bad place
for a decision rule.

What it does, in order:
  1. installs dependencies and clones the project repository;
  2. re-fetches every canon dataset and verifies its SHA-256 against the frozen
     registry — a run on bytes that are not the frozen bytes is not a valid run;
  3. loads one open-weight model at the revision pinned in `models.lock`, and
     refuses to continue if the hub served a different commit;
  4. runs the two mock controls *inside this session*, because a control that
     was only ever run somewhere else does not control anything here;
  5. runs the four memorization tests over the six canon datasets in English,
     writing every prompt and response to a JSONL log;
  6. prints the comparison against Bordt et al. and against our own GPT-4-0613
     numbers, and states the gate verdict.

Expected runtime: 1.5-3 hours for a 7-8B model in 4-bit on a T4.
"""

# %% [markdown]
# # Block A — validating the adapted pipeline
#
# Set the model below and Run All. The run is finished when the last cell prints
# a gate verdict; everything before that is setup and is meant to fail loudly.
#
# **This is not a hypothesis test.** No H1-H4 number may be quoted from this
# notebook. It answers one question: does our HF pipeline reproduce, in English,
# what the unmodified pipeline produced through the OpenAI API? Until it does,
# `TODO.md` blocks B-E stay closed.

# %% configuration
REPO_URL = "https://github.com/Lake3L/research_1_Memorization_In_RU_LLM.git"

# Block A runs the multilingual base model first: it is the control member of
# the base<->adapted pairs, so if the canon does not extract from *it*, nothing
# can be concluded from the Russian-adapted models either.
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

DATASET_GROUP = "canon"      # block A is the Western canon only
VARIANT = "raw"              # the published bytes, not a pandas round-trip
PROMPT_LANGUAGE = "en"       # block A is English only; RU is block D (H4)
LOAD_IN_4BIT = True
SEED = 42
SCALE = 1.0                  # 0.1 for a 10-minute smoke run of the whole plan

# %% [markdown]
# ## Install
#
# `pandas` is pinned below 3 because `tabmemcheck` breaks on pandas 3 — both
# `first_token_test` and `feature_completion_test` die (RESULTS_GATE.md §0), and
# a version bump would silently remove two of our four instruments.
#
# `transformers` is deliberately *not* pinned. Kaggle images ship their own
# torch, and forcing a transformers version against it is how a run dies at
# minute one. The backend handles both the 4.x and 5.x APIs, and the resolved
# version of everything is recorded in the results file, which is what §9
# actually requires: knowing what ran.

# %% install
import subprocess, sys, os

def sh(cmd):
    """Run a command and stream its output into the notebook cell.

    A subprocess writing to fd 1 lands in the kernel log, not in the cell, so a
    two-hour run would look like a hung notebook. Reading the pipe and printing
    from Python puts the progress where the person watching it is.
    """
    print(f"$ {cmd}", flush=True)
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True,
                               encoding="utf-8", errors="replace", bufsize=1)
    for line in process.stdout:
        print(line.rstrip(), flush=True)
    return process.wait()

sh(f"{sys.executable} -m pip install -q 'pandas<3' 'tabmemcheck==0.1.6' "
   f"transformers accelerate bitsandbytes jellyfish xgboost scipy")

if not os.path.exists("research_1_Memorization_In_RU_LLM"):
    sh(f"git clone -q {REPO_URL}")
os.chdir("research_1_Memorization_In_RU_LLM")
sh("git log -1 --format='repo commit: %H %ci'")
sys.path.insert(0, "src")

# %% [markdown]
# ## Data
#
# The CSVs are gitignored: they are other people's data and two of them are not
# ours to redistribute. `src/fetch_data.py` rebuilds them from their pinned
# sources and checks every file against the SHA-256 frozen in
# `AMENDMENT_1_DATASETS.md`. Five of the six canon files come from inside the
# installed `tabmemcheck` package, which is the same artefact the reference
# implementation tests against.

# %% fetch
sh(f"{sys.executable} src/fetch_data.py --group {DATASET_GROUP} "
   f"--report data/fetch_report_{DATASET_GROUP}.json")

import json
report = json.load(open(f"data/fetch_report_{DATASET_GROUP}.json", encoding="utf-8"))
missing = [r["dataset"] for r in report if r["status"] not in ("cached", "fetched")]
assert not missing, f"these datasets are not the frozen bytes: {missing}"
print(f"{len(report)} datasets verified against the freeze")

# %% [markdown]
# ## GPU check
#
# The plan is ~700 model calls. On CPU that is days, not hours, so a missing
# accelerator should stop the run here rather than at 3 a.m.

# %% gpu
import torch
print("cuda:", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no GPU")
assert torch.cuda.is_available(), "no GPU: enable an accelerator in the notebook settings"

# %% [markdown]
# ## The run
#
# `src/run_hf_gate.py` loads the model at its pinned revision, runs the mock
# controls in-session, executes the plan, and applies the gate rule that was
# written down before the run:
#
# * **PASS** — the mocks behave, and the header test passes on ≥2 of 6 canon
#   datasets or iris row completion beats the duplicate base rate at p<0.05.
# * **FAIL_ADAPTER** — the mocks behave but the model's answers do not even have
#   the shape of CSV rows. Diagnose the chat template and truncation.
# * **FAIL_NO_SIGNAL** — well-formed answers, nothing fires. A result about the
#   model, to be reported rather than tuned away (§10).

# %% run
cmd = (f"{sys.executable} -u src/run_hf_gate.py --model {MODEL_ID} "
       f"--group {DATASET_GROUP} --variant {VARIANT} --language {PROMPT_LANGUAGE} "
       f"--seed {SEED} --scale {SCALE}" + (" --load-in-4bit" if LOAD_IN_4BIT else ""))
status = sh(cmd)
# exit code 1 is a failed gate, not a crashed run: both write their results file
print("\nrun_hf_gate exit status:", status)

# %% [markdown]
# ## Collect the outputs
#
# Two files matter and both must come back: the counts, and the JSONL log of
# every prompt and response. The log is the more valuable of the two — a rented
# session is gone when it ends, and with the raw responses a scoring rule can be
# revised offline without paying for the run again.

# %% collect
import glob, shutil

produced = sorted(glob.glob("results/gateA_*.json")) + sorted(glob.glob("results/calls_*.jsonl"))
target = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
for path in produced:
    if os.path.abspath(os.path.dirname(path)) != os.path.abspath(target):
        shutil.copy(path, target)
    print(f"{os.path.getsize(path)/1e6:7.2f} MB  {path}")

latest = sorted(glob.glob("results/gateA_*.json"))[-1]
outcome = json.load(open(latest, encoding="utf-8"))
print("\ngate verdict:", outcome["gate"]["verdict"], "-", outcome["gate"]["reason"])
print("model revision loaded:", outcome["revision_loaded"])
print("\nsend back both files; they are committed to results/ and every number in")
print("RESULTS_GATE.md §5 is regenerated from them by a script, not typed in.")
