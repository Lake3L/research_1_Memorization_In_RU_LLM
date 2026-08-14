"""Kaggle/Colab runner for the memorization gate on open-weight models.

Built for block A — does the adapted pipeline reproduce the English result of the
unmodified one (PREREGISTRATION.md §8, second half) — and reused unchanged for the
runs that follow, since the plan, the scoring and the verdict rule are the same
and only the model list moves.

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
  3. loads each model in turn, in its own process, at the revision pinned in
     `models.lock`, refusing to continue if the hub served a different commit,
     and records where that model's chat template puts the system prompt;
  4. runs the two mock controls *inside this session*, because a control that
     was only ever run somewhere else does not control anything here;
  5. runs the four memorization tests over the six canon datasets in English,
     writing every prompt and response to a JSONL log;
  6. prints the comparison against Bordt et al. and against our own GPT-4-0613
     numbers, and states the gate verdict.

Expected runtime: about 1 h 15 min per 7B model in 4-bit on a T4x2 (measured),
so budget roughly two hours per 12B model and check the session limit before
queueing more than two.
"""

# %% [markdown]
# # Memorization gate runs on open-weight models
#
# Set the models below and Run All. The run is finished when the last cell prints
# a verdict per model; everything before that is setup and is meant to fail loudly.
#
# Runs now use the parameters Bordt et al. used for open models, read from their
# own code rather than from the library defaults, which differ: five few-shot
# blocks, eight prefix rows, header completion length 350, and the library's
# `max_tokens` instead of the caps we had inherited from a $5 OpenAI budget.
# `AMENDMENT_4_PROTOCOL_ALIGNMENT.md` §1 has the table of what changed and why.
#
# **This is still not a hypothesis test.** It runs one variant, one prompt
# language and one seed, where H1 asks for four variants and three seeds. What it
# produces is a decision about which probe H1 should use, and a first properly
# powered look at the base-vs-adapted direction.

# %% configuration
REPO_URL = "https://github.com/Lake3L/research_1_Memorization_In_RU_LLM.git"

# Each entry is one run in its own process, so the GPU is released between them.
# `extra` goes straight to src/run_hf_gate.py.
#
# This session is a 2x2: two models (a base and its Russian adaptation) x two
# prompting modes, over the same datasets at the same query counts. It is designed
# to answer two questions at once, and it is described in AMENDMENT_4.
#
#   Does the prompting mode matter?  In chat mode tabmemcheck sends a system
#     prompt plus five few-shot blocks drawn from OTHER datasets, then the prefix
#     rows. In completion mode it sends the prefix rows as raw text and nothing
#     else. Bordt et al. used completion mode for three of their five open models;
#     we have been using chat mode for everything, which may be why our signal is
#     so thin. Completion mode also removes the chat-template confound entirely,
#     because it uses no chat template.
#
#   Is the adaptation higher than its base?  The pilot said yes on every cell
#     where anything extracted, but at n=25-50 that is not decidable. Query counts
#     here come from src/power_h1b.py and exhaust iris and wine, which are bounded
#     by their own row counts.
#
# Completion-mode runs come first: they are the new information, and they are much
# faster because the prompts are a fraction of the size. If the session runs out of
# time, the two runs that matter will already be done.

PROBE = "--plan probe --protocol reference"

RUNS = [
    {"model": "mistralai/Mistral-Nemo-Instruct-2407",           "extra": f"{PROBE} --prompting completion"},
    {"model": "Vikhrmodels/Vikhr-Nemo-12B-Instruct-R-21-09-24", "extra": f"{PROBE} --prompting completion"},
    {"model": "mistralai/Mistral-Nemo-Instruct-2407",           "extra": f"{PROBE} --prompting chat"},
    {"model": "Vikhrmodels/Vikhr-Nemo-12B-Instruct-R-21-09-24", "extra": f"{PROBE} --prompting chat"},
]

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
# `AMENDMENT_1_DATASETS.md`. Each source is checked against that hash *before* it
# is accepted, so a source with the right rows and the wrong bytes falls through
# to the next one instead of being used — which is exactly what a local
# `tabmemcheck` checkout on Windows turned out to be (`AMENDMENT_2_LINE_ENDINGS.md`).

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
# The plan is 912 model calls per run, four runs. On CPU that is days, not hours,
# so a missing accelerator should stop the run here rather than at 3 a.m.
# Completion-mode prompts are a few hundred tokens; chat-mode ones a few thousand,
# so the first two runs are far quicker than the last two.

# %% gpu
import torch

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

assert torch.cuda.is_available(), "no GPU: enable an accelerator in the notebook settings"
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(f"gpu {i}: {props.name}, {props.total_memory / 1e9:.1f} GB, "
          f"compute capability {props.major}.{props.minor}"
          + ("  (Turing: no bfloat16)" if props.major == 7 else ""))

# Prove the 4-bit path works on a 1 GB model before downloading 24 GB of weights.
# The first 12B attempt died of OOM four minutes into the run because
# quantization had silently not taken effect; this costs a minute and settles it.
if LOAD_IN_4BIT:
    status = sh(f"{sys.executable} -u src/check_quantization.py")
    assert status == 0, ("4-bit loading does not work in this image — see the output "
                         "above. Running anyway would either OOM on a 12B model or "
                         "measure it in an unrecorded precision.")

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
#
# The whole model goes on GPU 0 (`--device-map single`). Spreading a quantized
# model over two cards is what killed the first 12B attempt: the placement was
# planned in one precision and the tensors arrived in another, and 9 GB of nf4
# weights managed to fill 29 GB of VRAM. One card is also the honest test —
# a 12B model fits there quantized and cannot fit there unquantized, so a
# configuration that silently failed to quantize now fails loudly instead.

# %% run
import glob, time

# Earlier runs are committed to results/ and arrive with the clone. Remember them,
# so the summary at the end reports this session rather than the whole history.
PRE_EXISTING = set(glob.glob("results/gateA_*.json")) | set(glob.glob("results/calls_*.jsonl"))

for run in RUNS:
    started = time.time()
    label = run["model"] + (f"   [{run['extra']}]" if run.get("extra") else "")
    print("\n" + "=" * 78 + f"\n{label}\n" + "=" * 78, flush=True)
    cmd = (f"{sys.executable} -u src/run_hf_gate.py --model {run['model']} "
           f"--group {DATASET_GROUP} --variant {VARIANT} --language {PROMPT_LANGUAGE} "
           f"--seed {SEED} --scale {SCALE}"
           + (" --load-in-4bit" if LOAD_IN_4BIT else "")
           + (" " + run["extra"] if run.get("extra") else ""))
    # exit code 1 is a failed gate, not a crashed run: both write their results file
    status = sh(cmd)
    print(f"\nexit status {status} after {(time.time() - started) / 60:.0f} min",
          flush=True)

# %% [markdown]
# ## Collect the outputs
#
# Two files matter and both must come back: the counts, and the JSONL log of
# every prompt and response. The log is the more valuable of the two — a rented
# session is gone when it ends, and with the raw responses a scoring rule can be
# revised offline without paying for the run again.

# %% collect
import shutil

produced = sorted((set(glob.glob("results/gateA_*.json"))
                   | set(glob.glob("results/calls_*.jsonl"))) - PRE_EXISTING)
print(f"{len(produced)} files produced by this session\n")
target = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
for path in produced:
    if os.path.abspath(os.path.dirname(path)) != os.path.abspath(target):
        shutil.copy(path, target)
    print(f"{os.path.getsize(path)/1e6:7.2f} MB  {path}")

for path in [p for p in produced if p.endswith(".json")]:
    outcome = json.load(open(path, encoding="utf-8"))
    template = (outcome.get("chat_template") or {}).get("system_position")
    header = [r for r in outcome["results"] if r.get("test") == "header"]
    passed = [r["dataset_key"] for r in header if r.get("verdict") == "pass"]
    fired = [f"{r['dataset_key']}/{r['test']} {r['matches']}/{r['n']}"
             for r in outcome["results"] if r.get("matches")]
    print(f"\n{outcome['model']}")
    print(f"   revision   : {outcome['revision_loaded']}")
    print(f"   template   : system prompt {template}")
    print(f"   placement  : system prompt {outcome.get('system_prompt_placement')}")
    print(f"   loaded     : {json.dumps(outcome.get('load', {}), ensure_ascii=False)[:110]}")
    print(f"   verdict    : {outcome['gate']['verdict']} — {outcome['gate']['reason']}")
    if not outcome["gate"].get("complete", True):
        print(f"   INCOMPLETE : {outcome['gate']['cells_run']}/"
              f"{outcome['gate']['cells_planned']} cells ran")
    print(f"   header pass: {len(passed)}/{len(header)} {passed}")
    print(f"   non-zero   : {fired or 'nothing'}")

print("\nSend back every gateA_*.json and calls_*.jsonl. The call log matters more than")
print("the counts: RESULTS_GATE.md is regenerated from it by src/rescore_calls.py, and a")
print("rented session cannot be re-run for free once it has ended.")
