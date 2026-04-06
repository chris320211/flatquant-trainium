#!/usr/bin/env python3
"""
Perplexity comparison: BF16 baseline vs FlatQuant+FP8 reparameterized weights.

Evaluates on wikitext-2 test set using standard HF (CPU, no Trainium needed).
Both models loaded as standard float16 AutoModelForCausalLM.

Usage:
  python eval_perplexity.py
"""

import torch
import math
import time
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# ── Config ────────────────────────────────────────────────────────────────────
BASE_MODEL        = "meta-llama/Llama-2-7b-hf"
REPARAM_MODEL     = "agent-workflow/tools/quantized_llama2_7b_reparameterized"
MAX_TOKENS        = 4096   # total tokens to evaluate (keep fast on CPU)
STRIDE            = 512    # sliding-window stride
SEQ_LEN           = 1024   # context window per chunk
DTYPE             = torch.float16
# ─────────────────────────────────────────────────────────────────────────────


def compute_perplexity(model, tokenizer, token_ids: torch.Tensor, seq_len: int, stride: int) -> float:
    """Sliding-window perplexity over a flat token tensor."""
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    n = token_ids.size(1)

    for begin in range(0, n - 1, stride):
        end = min(begin + seq_len, n)
        chunk = token_ids[:, begin:end]
        # Labels: shift by 1, mask tokens we've already scored in prev window
        target_len = end - max(begin, stride) if begin > 0 else end - 1
        if target_len <= 0:
            continue
        with torch.no_grad():
            out = model(chunk, labels=chunk)
            # out.loss is mean NLL over the full chunk; we want the tail `target_len` tokens
            # Re-compute manually to get per-token NLL for just the new tokens
            logits = out.logits  # [1, seq, vocab]
            shift_logits = logits[0, :-1, :]          # [seq-1, vocab]
            shift_labels = chunk[0, 1:]               # [seq-1]
            nll_per_tok = torch.nn.functional.cross_entropy(
                shift_logits, shift_labels, reduction='none'
            )
            # Only count the last `target_len` tokens (newly scored in this window)
            new_nll = nll_per_tok[-target_len:].sum().item()
            total_nll += new_nll
            total_tokens += target_len

        if end == n:
            break

    ppl = math.exp(total_nll / total_tokens)
    return ppl, total_tokens


def load_wikitext_tokens(tokenizer, max_tokens: int):
    """Load wikitext-2 test set and tokenize into a flat tensor."""
    print(f"  Loading wikitext-2 test set...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(dataset["text"])
    print(f"  Tokenizing (target: {max_tokens} tokens)...")
    ids = tokenizer(text, return_tensors="pt").input_ids
    ids = ids[:, :max_tokens]
    print(f"  Using {ids.size(1)} tokens")
    return ids


def evaluate_model(model_path: str, token_ids: torch.Tensor, label: str):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  Path: {model_path}")
    print(f"{'='*60}")

    t0 = time.time()
    print(f"  Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)

    print(f"  Loading model in float16 on CPU...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=DTYPE,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    load_time = time.time() - t0
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    print(f"  Loaded in {load_time:.1f}s  |  ~{param_bytes/1e9:.2f} GB in RAM")

    print(f"  Computing perplexity (seq_len={SEQ_LEN}, stride={STRIDE})...")
    t1 = time.time()
    ppl, n_tokens = compute_perplexity(model, tokenizer, token_ids, SEQ_LEN, STRIDE)
    eval_time = time.time() - t1
    print(f"  Done in {eval_time:.1f}s over {n_tokens} tokens")

    del model
    import gc; gc.collect()

    return ppl


def main():
    print("=" * 60)
    print("  PERPLEXITY COMPARISON: Baseline vs FlatQuant+FP8")
    print(f"  Dataset: wikitext-2 test, {MAX_TOKENS} tokens")
    print("=" * 60)

    # Load tokens once (reuse for both models, same tokenizer family)
    tok = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=False)
    token_ids = load_wikitext_tokens(tok, MAX_TOKENS)
    del tok

    results = {}

    results["BF16 Baseline (meta-llama/Llama-2-7b-hf)"] = evaluate_model(
        BASE_MODEL, token_ids, "BF16 Baseline"
    )

    results["FlatQuant+FP8 (reparameterized float16)"] = evaluate_model(
        REPARAM_MODEL, token_ids, "FlatQuant reparameterized (float16, pre-FP8)"
    )

    print(f"\n{'='*60}")
    print("  RESULTS SUMMARY")
    print(f"{'='*60}")
    baseline_ppl = None
    for name, ppl in results.items():
        marker = ""
        if baseline_ppl is None:
            baseline_ppl = ppl
        else:
            delta = ppl - baseline_ppl
            marker = f"  (+{delta:.2f} vs baseline)" if delta >= 0 else f"  ({delta:.2f} vs baseline)"
        print(f"  {name}")
        print(f"    PPL = {ppl:.4f}{marker}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
