"""
Single shared local Qwen fallback model.

Both semantic_risk_classifier.py and llm_reasoner.py import
from here instead of loading their own copy of the model,
so only ONE Qwen instance ever sits in RAM/VRAM.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from services.config import settings

LOCAL_MODEL_NAME = getattr(
    settings, "LOCAL_LLM_MODEL", "Qwen/Qwen2.5-1.5B-Instruct"
)

LOCAL_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_TOKENIZER = None
_MODEL = None


def _load():
    global _TOKENIZER, _MODEL

    if _TOKENIZER is not None and _MODEL is not None:
        return

    print(f"[LOCAL LLM] Loading fallback model: {LOCAL_MODEL_NAME}")
    print(f"[LOCAL LLM] Device: {LOCAL_DEVICE}")

    _TOKENIZER = AutoTokenizer.from_pretrained(LOCAL_MODEL_NAME)

    dtype = torch.float16 if LOCAL_DEVICE == "cuda" else torch.float32

    _MODEL = AutoModelForCausalLM.from_pretrained(
        LOCAL_MODEL_NAME, torch_dtype=dtype
    )

    _MODEL.to(LOCAL_DEVICE)
    _MODEL.eval()

    print("[LOCAL LLM] Qwen fallback loaded.")


def generate(system_prompt: str, user_prompt: str, max_new_tokens: int = 800) -> str:
    """
    Run one generation on the shared local Qwen model and
    return the raw decoded text (caller parses JSON etc).
    """

    _load()

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    formatted = _TOKENIZER.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = _TOKENIZER(
        formatted, return_tensors="pt", truncation=True, max_length=4096
    )
    inputs = {k: v.to(LOCAL_DEVICE) for k, v in inputs.items()}

    with torch.inference_mode():
        output_ids = _MODEL.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=_TOKENIZER.eos_token_id,
        )

    generated_ids = output_ids[0, inputs["input_ids"].shape[1]:]

    return _TOKENIZER.decode(generated_ids, skip_special_tokens=True).strip()
