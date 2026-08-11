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
    device_map: object = None
    log_path: Optional[str] = None

    model: object = field(default=None, repr=False)
    tokenizer: object = field(default=None, repr=False)
    loaded_revision: Optional[str] = None
    load_report: dict = field(default_factory=dict)
    n_calls: int = 0
    context: dict = field(default_factory=dict)  # tags written into every log line

    def __post_init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, revision=self.revision
        )
        kwargs = {"revision": self.revision}
        requested_dtype = self.dtype if self.dtype is not None else "auto"
        if self.quantization_config is not None:
            # `device_map` decides where the loader *plans* to put each module,
            # and planning across two GPUs is how a 12B model that occupies 9 GB
            # in nf4 came to fill 29 GB and die: the plan was made in one dtype
            # and the tensors materialised in another. bitsandbytes itself
            # defaults to a single device (`quantizer_bnb_4bit.update_device_map`),
            # and a single device is also the honest test of whether quantization
            # happened at all — a quantized 12B fits on one 16 GB card and an
            # unquantized one cannot, so a wrong configuration fails loudly here
            # instead of silently measuring a model in a precision we did not choose.
            kwargs.update(quantization_config=self.quantization_config,
                          device_map=self.device_map or {"": 0})
            # T4s are Turing and have no bfloat16. "auto" reads the checkpoint's
            # own dtype, which for these models is bfloat16, and that is both
            # slower and twice the memory of what we asked for.
            if requested_dtype == "auto":
                requested_dtype = torch.float16
        elif self.device_map:
            kwargs.update(device_map=self.device_map)
        try:  # transformers >= 5
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name, dtype=requested_dtype, **kwargs)
        except TypeError:  # transformers 4
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name, torch_dtype=requested_dtype, **kwargs)
        if self.quantization_config is None and not self.device_map:
            self.model = self.model.to(self.device)
        self.model.eval()
        self.loaded_revision = getattr(self.model.config, "_commit_hash", None)
        self.load_report = self._describe_load()
        if self.quantization_config is not None and not self.load_report["quantized"]:
            raise RuntimeError(
                "quantization was requested but the loaded model is not quantized "
                f"({self.load_report}). Refusing to continue: the run would silently "
                "measure a different precision than the one recorded.")
        if self.log_path:
            os.makedirs(os.path.dirname(os.path.abspath(self.log_path)), exist_ok=True)

    def _describe_load(self) -> dict:
        """What actually got loaded, in what precision, and how big it is.

        Recorded because the alternative is inferring it from a crash. The block A
        run claimed 4-bit in its filename and its arguments, and nothing in its
        output could confirm or deny that the weights were ever quantized.
        """
        report = {"quantized": bool(getattr(self.model, "hf_quantizer", None)),
                  "requested_quantization": self.quantization_config is not None}
        try:
            report["memory_footprint_gb"] = round(
                self.model.get_memory_footprint() / 1e9, 2)
        except Exception:
            report["memory_footprint_gb"] = None
        dtypes, devices = {}, set()
        for name, parameter in self.model.named_parameters():
            dtypes[str(parameter.dtype)] = dtypes.get(str(parameter.dtype), 0) + 1
            devices.add(str(parameter.device))
        report["parameter_dtypes"] = dtypes
        report["devices"] = sorted(devices)
        return report

    def chat_template_report(self) -> dict:
        """Where this model's template actually puts the system prompt.

        Not a formality. Mistral-Nemo's template moves the system message to
        immediately before the *final* user turn, while Qwen and Vikhr-Nemo put it
        first — and it does so silently, without raising, so the merge fallback
        below never fires and nothing in the counts would reveal it. Since H1b
        compares a base model against its Russian adaptation, and an adaptation
        can ship a different template than its base, a difference in where the
        instruction sits is a difference between the two arms that is not the
        thing under study. We do not force a common template — querying a model
        through the interface it was built for is the more faithful measurement —
        but the choice has to be visible in the record, so it is probed once at
        load and written into every results file.
        """
        probe = [{"role": "system", "content": "SYSTEM_MARKER"},
                 {"role": "user", "content": "USER_ONE"},
                 {"role": "assistant", "content": "ASSISTANT_ONE"},
                 {"role": "user", "content": "USER_TWO"}]
        report = {"accepts_system_role": None, "system_position": None}
        try:
            text = self.tokenizer.apply_chat_template(
                probe, add_generation_prompt=True, tokenize=False)
            report["accepts_system_role"] = True
        except Exception as e:
            report.update(accepts_system_role=False, error=f"{type(e).__name__}: {e}")
            return report
        where, first_user = text.find("SYSTEM_MARKER"), text.find("USER_ONE")
        report["system_position"] = ("before_first_user" if 0 <= where < first_user
                                     else "moved_to_last_user_turn" if where >= 0
                                     else "dropped")
        report["rendered"] = text
        return report

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
