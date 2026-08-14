"""Block A: the second half of the positive-control gate (PREREGISTRATION.md §8).

The first half is done (RESULTS_GATE.md): the unmodified pipeline reproduces
Bordt et al. through the OpenAI API. §8 also requires the *adapted* pipeline —
our HF backend, our dataset registry, our scoring — to reproduce the English
result of the unmodified one before it may be used for H1–H4. This module is
that run, and it is deliberately a plain script rather than notebook cells, so
that the same code path can be exercised locally against mocks before it is
handed a GPU.

Three things are separated on purpose, because a single number cannot tell them
apart:

  the instrument   — mock controls, run inside this very process, on this very
                     machine: a perfect memorizer must score ~100% and a
                     format-echo mock ~0%. If they do not, nothing else here is
                     evidence of anything.
  the adapter      — the shape of the model's answers (`well_formed_rate`).
                     A model that returns prose instead of CSV rows scores zero
                     for a reason that has nothing to do with memorization.
  the memorization — the counts, against the paper's and against our own
                     GPT-4-0613 reproduction.

Decision rule, fixed here before the run (§8 gives the criterion, this is its
executable form):

  PASS  if the mock controls behave, and the header test passes on at least two
        of the six canon datasets, or iris row completion beats the duplicate
        base rate at p < 0.05.
  FAIL — adapter   if the mocks behave but almost no answer is well-formed:
        diagnose the chat template and truncation, not the model.
  FAIL — no signal if the answers are well-formed and every count is zero. That
        is a finding about the model, not a broken pipeline, and §10 says to
        stop and report rather than to keep tuning prompts.

Usage:
  python src/run_hf_gate.py --mock perfect          # instrument check, no GPU
  python src/run_hf_gate.py --model Qwen/Qwen2.5-7B-Instruct --load-in-4bit
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np
import tabmemcheck as tabmem

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset_registry import load_registry, sha256  # noqa: E402
from mock_llm import MockAdapter, PerfectMemorizer, format_echo_mock  # noqa: E402
from prompt_language import set_language  # noqa: E402
from run_repro import PLANS, PAPER, PROTOCOL, dataset_key, run_one  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Our own GPT-4-0613 numbers from the first half of the gate (RESULTS_GATE.md
# §2), measured with this same code. They are the closer comparison than the
# paper's, because they share our sample sizes and our scoring.
OURS_GPT4 = {("iris.csv", "row"): (24, 25),
             ("openml-diabetes.csv", "feature"): (22, 25)}

# Bordt et al. Table 3 in words (p. 5): "there is evidence for the memorization
# of the most popular tabular datasets in almost all LLMs. However, with the
# exception of the Iris dataset, there is usually only evidence for the
# memorization of the initial rows and not of the entire dataset."
EXPECTATION = {
    "header": "passes on most canon datasets (Table 2/3)",
    "row": "fires mainly on iris; zero on adult and california-housing",
    "feature": "strong on diabetes and titanic; zero on adult and california-housing",
    "first_token": "above the mode baseline on iris",
}


def resolve_datasets(group="canon", variant="raw"):
    """Paths of the frozen files for one group, checked against their hashes.

    Verbatim memorization is a claim about specific bytes. A run against a file
    whose hash does not match the freeze is void, so this refuses to return one.
    """
    registry = load_registry()
    paths, problems = {}, []
    for name, rec in registry.items():
        if rec["group"] != group:
            continue
        info = rec.get("variants", {}).get(variant)
        if info is None:
            problems.append(f"{name}: no '{variant}' variant registered")
            continue
        path = os.path.join(ROOT, info["path"])
        if not os.path.exists(path):
            problems.append(f"{name}: missing {info['path']} — run src/fetch_data.py")
        elif sha256(path) != info["sha256"]:
            problems.append(f"{name}: {info['path']} does not match the frozen hash")
        else:
            paths[f"{name}.csv"] = path
    return paths, problems


def instrument_check(csv_path, seed=42, num_queries=10):
    """Run the two mocks through the real test, in this process, on this machine.

    RESULTS_GATE.md §0 did this before the paid run. It is repeated here rather
    than trusted, because the point of a control is that it travels with the
    experiment: a Kaggle image with a different pandas would break the counting
    silently, and only a control run in the same session would show it.
    """
    perfect = MockAdapter(PerfectMemorizer(csv_path))
    echo = MockAdapter(format_echo_mock)
    checks = {}
    for label, llm in (("perfect_memorizer", perfect), ("format_echo", echo)):
        suffixes, responses = tabmem.row_completion_test(
            csv_path, llm, num_queries=num_queries, rng=np.random.default_rng(seed),
            print_levenshtein=False)
        matches = sum(1 for s, r in zip(suffixes, responses) if s.strip() in r.strip())
        checks[label] = {"matches": matches, "n": len(responses)}
    ok = (checks["perfect_memorizer"]["matches"] >= 0.9 * num_queries
          and checks["format_echo"]["matches"] <= 0.1 * num_queries)
    checks["instrument_ok"] = bool(ok)
    return checks


def gate_verdict(results, instrument):
    """Apply the decision rule stated in the module docstring."""
    header_passes = [r for r in results
                     if r.get("test") == "header" and r.get("verdict") == "pass"]
    iris_row = next((r for r in results if r.get("dataset_key") == "iris.csv"
                     and r.get("test") == "row"), None)
    iris_significant = bool(iris_row and (iris_row.get("p_value") is not None)
                            and iris_row["p_value"] < 0.05)

    shaped = [r.get("well_formed_rate") for r in results
              if r.get("test") == "row" and r.get("well_formed_rate") is not None]
    well_formed = float(np.mean(shaped)) if shaped else None

    if not instrument.get("instrument_ok"):
        verdict, reason = "FAIL", "the mock controls did not behave: the counting code is wrong on this machine"
    elif len(header_passes) >= 2 or iris_significant:
        verdict, reason = "PASS", (f"{len(header_passes)}/6 header tests pass"
                                   + (", iris row completion significant" if iris_significant else ""))
    elif well_formed is not None and well_formed < 0.5:
        verdict, reason = "FAIL_ADAPTER", (
            f"only {well_formed:.0%} of row-completion answers even have the shape of a "
            "CSV row — diagnose the chat template, the system prompt and truncation")
    else:
        verdict, reason = "FAIL_NO_SIGNAL", (
            "answers are well-formed but no test fires: on this model the canon does not "
            "extract. Per §10 this is reported, not tuned away")

    # A verdict says what the cells that ran show. Whether they all ran is a
    # separate fact and has to travel with it: the 2026-08-13 pair each lost five
    # cells to CUDA OOM and still printed an unqualified PASS, which is a report
    # of a measurement that was a quarter missing.
    failed = [{"test": r.get("test"), "dataset": r.get("dataset_key"),
               "error": str(r.get("error"))[:120]}
              for r in results if "error" in r]
    return {"verdict": verdict, "reason": reason,
            "header_passes": len(header_passes),
            "iris_row_significant": iris_significant,
            "mean_well_formed_rate": round(well_formed, 4) if well_formed is not None else None,
            "cells_run": len(results) - len(failed), "cells_planned": len(results),
            "complete": not failed, "failed_cells": failed}


def comparison_table(results):
    lines = [f"{'test':12s} {'dataset':22s} {'ours':>10s} {'rate':>7s} "
             f"{'paper GPT-4':>12s} {'our GPT-4':>10s}  expectation",
             "-" * 104]
    for r in results:
        if "error" in r:
            lines.append(f"{r.get('test','?'):12s} {r.get('dataset_key','?'):22s} "
                         f"{'ERROR':>10s}  {r['error'][:60]}")
            continue
        key = (r["dataset_key"], r["test"])
        ref = PAPER.get(key, {})
        gpt4 = ref.get("gpt-4")
        ours4 = OURS_GPT4.get(key)
        if r["test"] == "header":
            got, rate = r.get("verdict", "?"), f"{r.get('rows_recovered', 0)}r"
            gpt4_text = str(ref.get("gpt-4", "n/a"))
        else:
            got = f"{r.get('matches','?')}/{r.get('n','?')}"
            rate = f"{r.get('rate', 0):.2f}"
            gpt4_text = f"{gpt4[0]}/{gpt4[1]}" if gpt4 else "n/a"
        lines.append(f"{r['test']:12s} {r['dataset_key']:22s} {got:>10s} {rate:>7s} "
                     f"{gpt4_text:>12s} {(f'{ours4[0]}/{ours4[1]}' if ours4 else 'n/a'):>10s}  "
                     f"{EXPECTATION.get(r['test'], '')}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--revision", default=None,
                    help="default: the revision pinned in models.lock")
    ap.add_argument("--mock", default="none", choices=["none", "perfect", "echo"],
                    help="run the plan against a mock instead of a model")
    ap.add_argument("--plan", default="gate_hf", choices=list(PLANS))
    ap.add_argument("--group", default="canon")
    ap.add_argument("--variant", default="raw")
    ap.add_argument("--language", default="en", choices=["en", "ru"])
    ap.add_argument("--load-in-4bit", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument("--device-map", default="single",
                    choices=["single", "auto", "balanced", "none"],
                    help="where the loader may place modules. 'single' keeps the "
                         "whole model on GPU 0, which is what bitsandbytes itself "
                         "defaults to and what makes a failed quantization fail "
                         "loudly instead of quietly spilling onto a second card")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--scale", type=float, default=1.0,
                    help="multiply every query count in the plan (for quick checks)")
    ap.add_argument("--prompting", default="chat", choices=["chat", "completion"],
                    help="'chat' sends a system prompt plus few-shot blocks from "
                         "other datasets; 'completion' sends the prefix rows as raw "
                         "text with neither. The authors used completion mode for "
                         "three of their five open models, and it is the probe of "
                         "AMENDMENT_4 §3 — it also removes the chat-template "
                         "confound, because it uses no chat template")
    ap.add_argument("--protocol", default="reference", choices=list(PROTOCOL),
                    help="'reference' uses the parameters Bordt et al. used for "
                         "open models (few_shot 5, 8 prefix rows, completion length "
                         "350, library max_tokens); 'legacy' reproduces what our own "
                         "runs before 2026-08-14 used. See AMENDMENT_4 §1")
    ap.add_argument("--system-prompt", default="template",
                    choices=["template", "first_user"],
                    help="'template' queries each model through its own chat "
                         "template, which is the default and the faithful choice. "
                         "'first_user' forces the system prompt to the front for "
                         "every model — the control for AMENDMENT_3 §3, where a "
                         "base and its adaptation place it differently")
    ap.add_argument("--only-cells", default=None,
                    help="comma-separated dataset:test pairs, e.g. "
                         "'uci-wine.csv:row,adult-train.csv:header'. Lets a session "
                         "repair the cells a previous one lost without paying for the "
                         "whole plan again")
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "results"))
    args = ap.parse_args()

    tabmem.config.print_prompts = False
    tabmem.config.print_responses = False
    set_language(args.language)

    paths, problems = resolve_datasets(args.group, args.variant)
    for p in problems:
        print(f"[data] {p}")
    if not paths:
        sys.exit("no dataset passed hash verification — nothing can be measured")
    print(f"[data] {len(paths)} datasets verified against the freeze: "
          f"{', '.join(sorted(paths))}")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    tag = (f"{args.plan}_{(args.mock if args.mock != 'none' else args.model).replace('/', '_')}"
           f"_{args.variant}_{args.language}"
           + ("" if args.prompting == "chat" else f"_{args.prompting}")
           + ("" if args.system_prompt == "template" else f"_sys-{args.system_prompt}")
           + f"_{stamp}")
    os.makedirs(args.out_dir, exist_ok=True)
    call_log = os.path.join(args.out_dir, f"calls_{tag}.jsonl")

    # ---------------------------------------------------------------- the model
    lock = {}
    lock_path = os.path.join(ROOT, "models.lock")
    if os.path.exists(lock_path):
        lock = json.load(open(lock_path, encoding="utf-8"))["models"]

    revision, loaded_revision, llm, template = args.revision, None, None, {}
    load_report = {}
    if args.mock != "none":
        # the perfect memorizer answers out of one specific CSV, so it is rebuilt
        # per dataset inside the run loop; one built on iris would score zero
        # everywhere else and look exactly like a real negative result
        llm = None if args.mock == "perfect" else MockAdapter(format_echo_mock)
        if llm is not None:
            llm.chat_mode = args.prompting == "chat"
        model_label = f"mock:{args.mock}"
    else:
        import torch
        from hf_llm import HFLLM
        if revision is None:
            revision = lock.get(args.model, {}).get("revision")
            if revision is None:
                print(f"[model] WARNING {args.model} is not pinned in models.lock; "
                      "a confirmatory run must not do this (§9)")
        quantization = None
        if args.load_in_4bit:
            from transformers import BitsAndBytesConfig
            quantization = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        device_map = ({"": 0} if args.device_map == "single"
                      else None if args.device_map == "none" else args.device_map)
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                total = torch.cuda.get_device_properties(i).total_memory / 1e9
                print(f"[gpu {i}] {torch.cuda.get_device_name(i)} — {total:.1f} GB")
        print(f"[model] loading {args.model} rev={revision} on {device} "
              f"{'in 4-bit' if args.load_in_4bit else 'unquantized'}, "
              f"device_map={device_map}")
        llm = HFLLM(model_name=args.model, revision=revision, device=device,
                    quantization_config=quantization, device_map=device_map,
                    chat_mode=(args.prompting == "chat"),
                    system_prompt_placement=args.system_prompt, log_path=call_log)
        print(f"[model] loaded: {json.dumps(llm.load_report, ensure_ascii=False)}")
        loaded_revision = llm.loaded_revision
        model_label = args.model
        if revision and loaded_revision and loaded_revision != revision:
            sys.exit(f"[model] loaded {loaded_revision}, models.lock says {revision} — "
                     "refusing to measure weights we did not pin")
        print(f"[model] loaded revision {loaded_revision}")
        load_report = llm.load_report
        template = llm.chat_template_report()
        print(f"[model] chat template: system role "
              f"{'accepted' if template.get('accepts_system_role') else 'REJECTED (merged into the first user turn)'}"
              f", position: {template.get('system_position')}")

    # ------------------------------------------------------------ the instrument
    print("\n[instrument] mock controls, in this process")
    instrument = instrument_check(paths["iris.csv"], seed=args.seed)
    print(f"[instrument] perfect memorizer "
          f"{instrument['perfect_memorizer']['matches']}/{instrument['perfect_memorizer']['n']}, "
          f"format echo {instrument['format_echo']['matches']}/{instrument['format_echo']['n']} "
          f"-> {'OK' if instrument['instrument_ok'] else 'BROKEN'}")

    # -------------------------------------------------------------------- the run
    plan = PLANS[args.plan]
    if args.only_cells:
        wanted = {tuple(pair.split(":")) for pair in args.only_cells.split(",")}
        plan = [cell for cell in plan if (cell[0], cell[1]) in wanted]
        if not plan:
            sys.exit(f"--only-cells matched nothing in plan '{args.plan}'")
    print(f"\n[run] plan '{args.plan}', variant '{args.variant}', prompts "
          f"'{args.language}', {len(plan)} cells")
    results = []
    for csv_name, test, num_queries in plan:
        path = paths.get(csv_name)
        if path is None:
            print(f"  {csv_name:24s} {test:12s} skipped (not in group '{args.group}')")
            continue
        num_queries = max(1, int(round(num_queries * args.scale)))
        if args.mock == "perfect":
            llm = MockAdapter(PerfectMemorizer(path))
            llm.chat_mode = args.prompting == "chat"
        if hasattr(llm, "context"):
            llm.context = {"dataset": csv_name, "test": test,
                           "variant": args.variant, "prompt_language": args.language}
        # a cell is minutes of silence on a GPU; say what is starting, not only
        # what finished, so a stalled run is distinguishable from a slow one
        # the header test always uses its four splits and ignores num_queries
        shown = "4 splits" if test == "header" else f"n={num_queries}"
        print(f"  {csv_name:24s} {test:12s} {shown:<9s} ...", flush=True)
        if hasattr(llm, "free_memory"):
            llm.free_memory()
        started = time.time()
        try:
            r = run_one(llm, path, test, num_queries, args.seed,
                        protocol=args.protocol)
        except Exception as e:  # one broken cell must not kill the run
            r = {"dataset": path, "dataset_key": dataset_key(path), "test": test,
                 "error": f"{type(e).__name__}: {e}"}
        r.update(variant=args.variant, prompt_language=args.language,
                 model=model_label, revision_requested=revision,
                 revision_loaded=loaded_revision,
                 seconds=round(time.time() - started, 1))
        if hasattr(llm, "peak_memory_gb"):
            r["peak_memory_gb"] = llm.peak_memory_gb()
        results.append(r)
        summary = (f"{r.get('matches')}/{r.get('n')}" if "matches" in r
                   else r.get("verdict", r.get("error", "?")))
        extra = (f"  well-formed {r['well_formed_rate']:.0%}, "
                 f"near-match {r['near_match_rate']:.0%}"
                 if r.get("well_formed_rate") is not None else "")
        print(f"  {csv_name:24s} {test:12s} -> {str(summary):>10s}{extra}"
              f"  [{r['seconds']:.0f}s]", flush=True)

    # ------------------------------------------------------------------ the verdict
    verdict = gate_verdict(results, instrument)
    if args.plan != "gate_hf":
        # The §8 gate rule is defined over the gate plan. Any other plan sees a
        # different set of cells, so calling its outcome a gate verdict would be
        # a claim about tests that were never run.
        verdict = {**verdict, "verdict": f"DESCRIPTIVE ({args.plan})",
                   "reason": f"plan '{args.plan}' is not the §8 gate plan; the "
                             "counts stand on their own and no gate verdict is implied"}
    if args.only_cells:
        # A repair run sees a subset by construction, so its "verdict" would be a
        # statement about cells that were deliberately not run. Say so instead.
        verdict = {"verdict": "PARTIAL", "reason": f"repair run of {len(plan)} cells; "
                   "the gate verdict comes from the full plan, not from this",
                   "repaired_cells": args.only_cells, **{k: v for k, v in verdict.items()
                                                         if k not in ("verdict", "reason")}}
    print("\n" + comparison_table(results))
    print(f"\nGATE (block A): {verdict['verdict']} — {verdict['reason']}")
    if not verdict["complete"]:
        print(f"INCOMPLETE: {verdict['cells_run']}/{verdict['cells_planned']} cells ran. "
              "The verdict describes only those; the plan was not finished.")
        for cell in verdict["failed_cells"]:
            print(f"   missing: {cell['dataset']} {cell['test']} — {cell['error'][:70]}")
        repair = ",".join(f"{c['dataset']}:{c['test']}" for c in verdict["failed_cells"])
        print(f"   repair with: --only-cells {repair}")

    out = {
        "run": tag, "timestamp_utc": stamp, "block": "A",
        "model": model_label, "revision_requested": revision,
        "revision_loaded": loaded_revision,
        "plan": args.plan, "group": args.group, "variant": args.variant,
        "prompt_language": args.language, "seed": args.seed, "scale": args.scale,
        "system_prompt_placement": args.system_prompt,
        "prompting": args.prompting, "protocol": args.protocol,
        "protocol_settings": {k: v for k, v in PROTOCOL[args.protocol].items()},
        "datasets": {k: os.path.relpath(v, ROOT).replace(os.sep, "/")
                     for k, v in paths.items()},
        "versions": versions(),
        "chat_template": template,
        "load": load_report,
        "instrument_check": instrument,
        "gate": verdict,
        "results": results,
    }
    out_path = os.path.join(args.out_dir, f"gateA_{tag}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"wrote {out_path}")
    if os.path.exists(call_log):
        print(f"wrote {call_log} ({sum(1 for _ in open(call_log, encoding='utf-8'))} calls)")
    return 0 if verdict["verdict"] == "PASS" else 1


def versions():
    """What actually ran. §9 pins requirements; a rented image may ignore them."""
    import platform
    from importlib import metadata
    out = {"python": platform.python_version(), "platform": platform.platform()}
    for package in ("numpy", "pandas", "scipy", "torch", "transformers",
                    "tabmemcheck", "jellyfish", "accelerate", "bitsandbytes"):
        try:  # not every package exposes __version__, but all expose metadata
            out[package] = metadata.version(package)
        except Exception:
            out[package] = None
    return out


if __name__ == "__main__":
    sys.exit(main())
