"""Kaggle/Colab runner for the memorization tests on open-weight models.

Kept as a .py file and converted to .ipynb by `src/build_notebook.py`, so that the
pipeline stays reviewable in git rather than buried in JSON cell arrays.

The notebook is a driver and nothing more. Everything that decides anything lives
in `src/`, is exercised against mocks without a GPU before it is used, and is
configured from `notebooks/session.json`. What remains here is what genuinely
needs an accelerator: install, fetch, load the weights, run, collect.

In order:
  1. install the pinned dependencies;
  2. update the repository checkout and read the session definition;
  3. rebuild every dataset and verify its SHA-256 against the frozen registry;
  4. confirm the hardware and that 4-bit loading works in this image;
  5. run each model at its pinned revision, logging every prompt and response;
  6. collect the outputs.

Runtime: roughly 25 minutes per completion-mode run and 70 per chat-mode run for a
12B model in 4-bit on a T4, plus a few minutes for the weights.
"""

# %% [markdown]
# # Memorization tests on open-weight models
#
# **Run All. Nothing in this notebook needs editing.**
#
# What this session runs — the models, the plan, the prompting mode, the dataset
# group and the seed — is defined in `notebooks/session.json` and read from the
# repository below. To change the experiment, edit that file and push.
#
# Every cell before the run is a precondition and is meant to stop the notebook if
# it is not met. The run is finished when the last cell prints a summary per model.
#
# The measurement code is unmodified `tabmemcheck` 0.1.6 with the parameters Bordt
# et al. used for open-weight models: five few-shot blocks, eight prefix rows, a
# header completion length of 350. See `AMENDMENT_4_PROTOCOL_ALIGNMENT.md`.

# %% configuration
REPO_URL = "https://github.com/Lake3L/research_1_Memorization_In_RU_LLM.git"

# %% [markdown]
# ## Install
#
# `pandas` is held below 3: `tabmemcheck` relies on behaviour that changed there,
# and both `first_token_test` and `feature_completion_test` stop working, which
# would silently remove two of the four instruments.
#
# `transformers` is deliberately unpinned. Hosted images ship their own torch, and
# pinning against it is a frequent cause of an environment that will not resolve.
# The backend supports both the 4.x and 5.x APIs, and the resolved version of every
# package is recorded in the results file — which is what reproducibility requires:
# knowing exactly what ran.

# %% install
import subprocess, sys, os

def sh(cmd):
    """Run a command and stream its output into the notebook cell.

    A subprocess writing to fd 1 reaches the kernel log rather than the cell, so a
    long run would appear to hang. Reading the pipe and printing from Python puts
    the progress where it can be seen.
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

# %% [markdown]
# ## Repository and session
#
# The working directory persists between sessions, so the checkout is fetched and
# hard-reset rather than cloned conditionally: the code that runs is always the
# code at `origin/master`, and its commit is printed.
#
# The session definition is then read from that checkout, and the cell verifies
# that the code understands the flags and plans the session asks for before any
# weights are downloaded.

# %% repository
if not os.path.exists("research_1_Memorization_In_RU_LLM"):
    sh(f"git clone -q {REPO_URL}")
os.chdir("research_1_Memorization_In_RU_LLM")
sh("git fetch -q origin && git reset -q --hard origin/master && git clean -qfd src notebooks")
sh("git log -1 --format='REPO COMMIT: %H  %ci'")
sys.path.insert(0, "src")

import json
SESSION = json.load(open("notebooks/session.json", encoding="utf-8"))
DATASET_GROUP = SESSION["dataset_group"]
VARIANT = SESSION["variant"]
PROMPT_LANGUAGE = SESSION["prompt_language"]
LOAD_IN_4BIT = SESSION["load_in_4bit"]
SEED, SCALE = SESSION["seed"], SESSION["scale"]
RUNS = SESSION["runs"]

print(f"\nSESSION {SESSION['session_id']}: {len(RUNS)} runs")
print(SESSION["purpose"])
for r in RUNS:
    print(f"   {r['model']}  {r['extra']}")

help_text = subprocess.run([sys.executable, "src/run_hf_gate.py", "--help"],
                           capture_output=True, text=True).stdout
missing = [f for f in SESSION["requires"]["flags"] if f not in help_text]
assert not missing, (f"the checkout does not support {missing}; it is behind the "
                     f"session definition")
for plan in SESSION["requires"]["plans"]:
    assert plan in help_text, f"plan '{plan}' unknown to the checkout"
print("\ncheckout supports every flag and plan this session requires")

# %% [markdown]
# ## Data
#
# The CSVs are not committed — they are other people's data, and two of them carry
# licences that do not permit redistribution. `src/fetch_data.py` rebuilds them
# from pinned sources and checks each against the SHA-256 frozen in
# `AMENDMENT_1_DATASETS.md`.
#
# The hash is verified *while a source is being chosen*, not after: a source can
# return a file with the right rows and the wrong bytes, and verbatim memorization
# is a claim about bytes. A source that does not match is rejected and the next one
# is tried; a dataset that matches none stops the run.

# %% fetch
sh(f"{sys.executable} src/fetch_data.py --group {DATASET_GROUP} "
   f"--report data/fetch_report_{DATASET_GROUP}.json")

report = json.load(open(f"data/fetch_report_{DATASET_GROUP}.json", encoding="utf-8"))
missing = [r["dataset"] for r in report if r["status"] not in ("cached", "fetched")]
assert not missing, f"these datasets are not the frozen bytes: {missing}"
print(f"{len(report)} datasets verified against the freeze")

# %% [markdown]
# ## Hardware
#
# Two preconditions. An accelerator has to be present — these plans are hundreds
# of model calls each, which is days of CPU time. And 4-bit loading has to work,
# which is confirmed on a 1 GB model first: a large model whose quantization did
# not take effect either exhausts the card or is measured in a precision other
# than the one recorded.
#
# Compute capability is printed because Turing cards have no bfloat16, and the
# backend selects float16 accordingly.

# %% gpu
import torch

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

assert torch.cuda.is_available(), "no GPU: enable an accelerator in the notebook settings"
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(f"gpu {i}: {props.name}, {props.total_memory / 1e9:.1f} GB, "
          f"compute capability {props.major}.{props.minor}"
          + ("  (Turing: no bfloat16)" if props.major == 7 else ""))

if LOAD_IN_4BIT:
    status = sh(f"{sys.executable} -u src/check_quantization.py")
    assert status == 0, ("4-bit loading is not working in this image; see the output "
                         "above. Continuing would either exhaust the card or measure "
                         "the model in an unrecorded precision.")

# %% [markdown]
# ## The run
#
# For each model, `src/run_hf_gate.py` loads it at the revision pinned in
# `models.lock` and refuses to continue if the hub served a different commit; runs
# the two mock controls inside the session, because a control that ran elsewhere
# controls nothing here; executes the plan; and applies the decision rule fixed
# before the run:
#
# * **PASS** — the mock controls behave, and the header test passes on ≥2 of 6
#   canon datasets or iris row completion beats the duplicate base rate at p<0.05.
# * **FAIL_ADAPTER** — the controls behave but the answers do not have the shape of
#   CSV rows, which points at the chat template or truncation rather than the model.
# * **FAIL_NO_SIGNAL** — well-formed answers and nothing fires. That is a result
#   about the model and is reported as one.
#
# The whole model is placed on one device. A quantized 12B model fits on a single
# 16 GB card and an unquantized one does not, so single-device placement makes the
# precision self-evident instead of something to be inferred later.

# %% run
import glob, time

# results/ arrives with the clone, so the summary below reports this session only
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
    # exit status 1 means the decision rule was not met, not that the run crashed;
    # both write their results file
    status = sh(cmd)
    print(f"\nexit status {status} after {(time.time() - started) / 60:.0f} min",
          flush=True)

# %% [markdown]
# ## Outputs
#
# Two files per run, and both are needed: the counts, and the JSONL log of every
# prompt and response. The log is the more valuable of the two. A hosted session
# ends and takes its state with it, and with the raw responses any scoring rule can
# be revised offline without running the models again.

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
    print(f"\n{outcome['model']}  [{outcome.get('prompting')}]")
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

print("\nSend back every gateA_*.json and calls_*.jsonl listed above.")
