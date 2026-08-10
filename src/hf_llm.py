"""HuggingFace backend for tabmemcheck (stage 1 of the adaptation plan).

Implements tabmemcheck.LLM_Interface for local HF models so that all
memorization tests run without any OpenAI/Anthropic API. Test logic and
success criteria in tabmemcheck are untouched — this is a backend only.

Design constraints (see PREREGISTRATION.md §2, §3, §9):
- temperature 0 maps to greedy decoding (deterministic on fixed hardware);
- model revisions are pinned by the caller (models.lock), and the commit hash
  actually loaded is read back from the model config and logged, because a
  pinned name is not evidence of the weights that ran;
- chat templates without a system role get the system message merged into
  the first user turn (several Russian models need this fallback);
- every call is written to a JSONL log. A confirmatory run happens on rented or
  free compute that we do not get to keep; if only the counts came back, a
  scoring rule could never be revised without paying for the run again.

Version note. `apply_chat_template` returns a plain tensor under transformers 4
and a BatchEncoding under transformers 5 (`return_dict` defaults to True there),
and `torch_dtype` became `dtype`. Both are handled, because Kaggle and Colab
images move faster than our pins do.
"""

import json
import os
import time
from dataclasses import dataclass, field
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import tabmemcheck


@dataclass
class HFLLM(tabmemcheck.LLM_Interface):
    model_name: str = ""
    revision: Optional[str] = None
    device: str = "cpu"
    dtype: Optional[torch.dtype] = None
    chat_mode: bool = True
    quantization_config: object = None
    log_path: Optional[str] = None

    model: object = field(default=None, repr=False)
    tokenizer: object = field(default=None, repr=False)
    loaded_revision: Optional[str] = None
    n_calls: int = 0
    context: dict = field(default_factory=dict)  # tags written into every log line

    def __post_init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, revision=self.revision
        )
        kwargs = {"revision": self.revision}
        requested_dtype = self.dtype if self.dtype is not None else "auto"
        if self.quantization_config is not None:
            # accelerate places a quantized model itself; moving it afterwards fails
            kwargs.update(quantization_config=self.quantization_config,
                          device_map="auto")
        try:  # transformers >= 5
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name, dtype=requested_dtype, **kwargs)
        except TypeError:  # transformers 4
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name, torch_dtype=requested_dtype, **kwargs)
        if self.quantization_config is None:
            self.model = self.model.to(self.device)
        self.model.eval()
        self.loaded_revision = getattr(self.model.config, "_commit_hash", None)
        if self.log_path:
            os.makedirs(os.path.dirname(os.path.abspath(self.log_path)), exist_ok=True)

    # ------------------------------------------------------------------ logging

    def _log(self, kind, payload, response, temperature, max_tokens, n_input, seconds):
        self.n_calls += 1
        if not self.log_path:
            return
        record = {
            "i": self.n_calls,
            "t": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "model": self.model_name,
            "revision_requested": self.revision,
            "revision_loaded": self.loaded_revision,
            "kind": kind,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "n_input_tokens": n_input,
            "seconds": round(seconds, 2),
            kind: payload,
            "response": response,
            **self.context,
        }
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # --------------------------------------------------------------- generation

    def _generate(self, encoded, temperature: float, max_tokens: int) -> tuple:
        """Greedy at temperature 0. Returns (text, n_input_tokens)."""
        # transformers 5 returns a BatchEncoding here, transformers 4 a bare
        # tensor. BatchEncoding is a UserDict, so `isinstance(..., dict)` is
        # False for it — duck-type on the mapping instead.
        if hasattr(encoded, "keys") and "input_ids" in encoded:
            input_ids = encoded["input_ids"]
            attention_mask = encoded.get("attention_mask")
        else:
            input_ids = encoded
            attention_mask = torch.ones_like(input_ids)
        input_ids = input_ids.to(self.model.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.model.device)

        do_sample = temperature > 0
        with torch.no_grad():
            out = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_tokens,
                do_sample=do_sample,
                # the model's own generation_config carries sampling defaults
                # (Qwen ships temperature 0.7 / top_p 0.8); override them rather
                # than rely on them being ignored when do_sample is False
                temperature=temperature if do_sample else None,
                top_p=None,
                top_k=None,
                pad_token_id=self.tokenizer.pad_token_id
                or self.tokenizer.eos_token_id,
            )
        new_tokens = out[0][input_ids.shape[1]:]
        return (self.tokenizer.decode(new_tokens, skip_special_tokens=True),
                int(input_ids.shape[1]))

    def completion(self, prompt, temperature, max_tokens):
        started = time.time()
        encoded = self.tokenizer(prompt, return_tensors="pt")
        text, n_input = self._generate(encoded, temperature, max_tokens)
        self._log("prompt", prompt, text, temperature, max_tokens, n_input,
                  time.time() - started)
        return text

    def chat_completion(self, messages, temperature, max_tokens):
        started = time.time()
        try:
            encoded = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt"
            )
        except Exception:
            # chat template rejects the system role: merge it into the first
            # user message and retry
            merged = []
            system = ""
            for m in messages:
                if m["role"] == "system":
                    system = m["content"]
                elif m["role"] == "user" and system:
                    merged.append(
                        {"role": "user", "content": system + "\n\n" + m["content"]}
                    )
                    system = ""
                else:
                    merged.append(m)
            encoded = self.tokenizer.apply_chat_template(
                merged, add_generation_prompt=True, return_tensors="pt"
            )
        text, n_input = self._generate(encoded, temperature, max_tokens)
        self._log("messages", messages, text, temperature, max_tokens, n_input,
                  time.time() - started)
        return text


def hf_setup(model_name: str, revision: Optional[str] = None,
             device: str = "cpu", chat_mode: bool = True,
             log_path: Optional[str] = None) -> HFLLM:
    return HFLLM(model_name=model_name, revision=revision, device=device,
                 chat_mode=chat_mode, log_path=log_path)
