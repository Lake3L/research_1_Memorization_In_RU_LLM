"""Smoke test of the HF backend: does the adapter itself work, end to end.

Runs the header, row-completion and feature-completion tests against a 0.5B
model on CPU. This is *not* a memorization measurement — a 0.5B model is not
expected to reproduce anything from Table 5 — it is the check that the plumbing
between tabmemcheck and a local HF model is intact: chat templating, greedy
decoding, the transformers-4/5 differences in `apply_chat_template`, and the
per-call JSONL log that a confirmatory run depends on.

Run this before any paid or rented compute:

  python src/smoke_hf_header.py            # ~10 minutes on CPU
  python src/smoke_hf_header.py --model Qwen/Qwen2.5-0.5B-Instruct --queries 3
"""

import argparse
import json
import os
import sys

import numpy as np
import tabmemcheck as tabmem

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hf_llm import HFLLM  # noqa: E402
from metrics import header_verdict  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--csv", default=os.path.join(ROOT, "data", "canon", "iris.csv"))
    ap.add_argument("--queries", type=int, default=3)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--log", default=os.path.join(ROOT, "results", "smoke_hf_calls.jsonl"))
    args = ap.parse_args()

    tabmem.config.print_prompts = False
    tabmem.config.print_responses = False
    tabmem.config.max_tokens = 300

    if os.path.exists(args.log):
        os.remove(args.log)

    print(f"loading {args.model} on {args.device} ...", flush=True)
    llm = HFLLM(model_name=args.model, revision=args.revision, device=args.device,
                log_path=args.log)
    llm.context = {"phase": "smoke"}
    print(f"loaded. revision on disk: {llm.loaded_revision}", flush=True)
    print(f"chat template present: {bool(getattr(llm.tokenizer, 'chat_template', None))}",
          flush=True)

    checks = {}

    print("\n[1/4] header test", flush=True)
    _, completion, response = tabmem.header_test(
        args.csv, llm, rng=np.random.default_rng(42), verbose=False)
    verdict = header_verdict(completion, response)
    print(f"      prefix match {verdict['prefix_match_chars']} chars, "
          f"{verdict['rows_recovered']} full rows recovered -> {verdict['verdict']}")
    print(f"      response starts: {response[:70]!r}")
    checks["header_returned_text"] = len(response) > 0

    print(f"\n[2/4] row completion ({args.queries} queries)", flush=True)
    suffixes, responses = tabmem.row_completion_test(
        args.csv, llm, num_queries=args.queries, rng=np.random.default_rng(42),
        print_levenshtein=False)
    print(f"      example response: {responses[0][:70]!r}")
    checks["row_all_answered"] = len(responses) == args.queries
    checks["row_nonempty"] = all(len(r.strip()) > 0 for r in responses)

    print(f"\n[3/4] feature completion ({args.queries} queries)", flush=True)
    values, feature_responses = tabmem.feature_completion_test(
        args.csv, llm, feature_name="petal_length", num_queries=args.queries,
        rng=np.random.default_rng(42))
    print(f"      truth {values[:3]} vs model {feature_responses[:3]}")
    checks["feature_all_answered"] = len(feature_responses) == args.queries

    print("\n[4/4] call log", flush=True)
    lines = [json.loads(l) for l in open(args.log, encoding="utf-8")]
    print(f"      {len(lines)} calls logged to {os.path.relpath(args.log, ROOT)}")
    checks["log_matches_call_count"] = len(lines) == llm.n_calls
    checks["log_has_prompt_and_response"] = all(
        ("messages" in l or "prompt" in l) and "response" in l for l in lines)
    checks["log_has_revision"] = all("revision_loaded" in l for l in lines)
    checks["log_has_token_counts"] = all(l.get("n_input_tokens", 0) > 0 for l in lines)

    print("\nadapter checks:")
    for name, ok in checks.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    failed = [n for n, ok in checks.items() if not ok]
    print("\nSMOKE TEST:", "PASSED" if not failed else f"FAILED ({', '.join(failed)})")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
