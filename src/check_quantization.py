"""Does 4-bit loading actually work in this environment? Ask a 1 GB model first.

The 12B run of 2026-08-12 spent four minutes downloading weights and then died of
CUDA OOM because `quantization_config` had not taken effect: the model was being
materialised in bfloat16 while the placement had been planned for nf4. The cost of
finding that out was a session; the cost of finding it out here is about a minute,
because a 0.5B model exercises exactly the same code path.

Checks, in order:
  1. bitsandbytes imports and CUDA is visible;
  2. a small model loads with the same `BitsAndBytesConfig` the real run uses;
  3. the loaded model reports itself as quantized — not merely that no error was
     raised, which is what the earlier configuration also managed;
  4. its memory footprint is what nf4 implies rather than what fp16 implies;
  5. it still generates text, so the quantized path is usable and not just loadable.

Exit code 0 means the real run may proceed. Anything else means it must not.

Usage:
  python src/check_quantization.py
  python src/check_quantization.py --model Qwen/Qwen2.5-0.5B-Instruct
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--max-footprint-gb", type=float, default=0.8,
                    help="0.5B params is ~1.0 GB in fp16 and ~0.4 GB in nf4, so a "
                         "footprint above this means the weights were not quantized")
    args = ap.parse_args()

    import torch
    if not torch.cuda.is_available():
        print("[preflight] no CUDA — nothing to check, and nothing can run")
        return 2
    try:
        import bitsandbytes
        print(f"[preflight] bitsandbytes {bitsandbytes.__version__}")
    except Exception as e:
        print(f"[preflight] bitsandbytes unusable: {type(e).__name__}: {e}")
        return 3

    from transformers import BitsAndBytesConfig
    from hf_llm import HFLLM

    quantization = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)

    print(f"[preflight] loading {args.model} in 4-bit on a single device ...")
    try:
        llm = HFLLM(model_name=args.model, device="cuda",
                    quantization_config=quantization, device_map={"": 0})
    except Exception as e:
        print(f"[preflight] FAILED to load: {type(e).__name__}: {e}")
        return 4

    report = llm.load_report
    print(f"[preflight] {json.dumps(report, ensure_ascii=False)}")

    problems = []
    if not report["quantized"]:
        problems.append("the model is not quantized despite the request")
    footprint = report.get("memory_footprint_gb")
    if footprint is not None and footprint > args.max_footprint_gb:
        problems.append(f"footprint {footprint} GB looks unquantized "
                        f"(expected under {args.max_footprint_gb} GB)")

    answer = llm.chat_completion(
        [{"role": "user", "content": "Reply with the single word: ready"}], 0, 8)
    print(f"[preflight] generation returned {answer.strip()[:40]!r}")
    if not answer.strip():
        problems.append("the quantized model loads but generates nothing")

    if problems:
        print("\n[preflight] DO NOT RUN — " + "; ".join(problems))
        print("[preflight] a large model would either OOM or be measured in a "
              "precision we did not choose and did not record.")
        return 1
    print("\n[preflight] 4-bit works here; the real run may proceed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
