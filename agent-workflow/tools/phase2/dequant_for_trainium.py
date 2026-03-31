#!/usr/bin/env python3
"""
Dequantize FlatQuant INT4 model to BF16 for Trainium2 tracing.

FlatQuant INT4 weights won't trace cleanly through XLA, so we need to dequantize
to BF16 first. This trades INT4 compression benefits for Trainium2 compatibility.

Usage:
    python dequant_for_trainium.py --quantized_model ./quantized_llama2_7b --output ./llama2_bf16_for_trainium
"""

import sys
import torch
from pathlib import Path
from typing import Dict

# CRITICAL: Import transformers FIRST, before any FlatQuantBundled modules
# This prevents FlatQuantBundled/deploy/transformers from shadowing the real one
from transformers import AutoModelForCausalLM

# NOTE: FlatQuantBundled should already be in PYTHONPATH from setup_env.sh
# DO NOT add it again to sys.path - that will cause the shadowing issue


def dequantize_flatquant_model(quantized_path: str, output_path: str) -> bool:
    """
    Convert FlatQuant INT4 model to BF16 for Trainium tracing.

    Args:
        quantized_path: Path to quantized model checkpoint (from Phase 1)
        output_path: Path to save BF16 model

    Returns:
        True if successful, False otherwise
    """
    print("=" * 60)
    print("FlatQuant INT4 → BF16 Dequantization for Trainium2")
    print("=" * 60)

    try:
        from llama_2_7b_hf_utils import FlatQuantLlamaMLP, FlatQuantLlamaAttention

        # Step 1: Load quantized model
        print(f"\n[1/3] Loading quantized model from {quantized_path}")
        model_quantized = AutoModelForCausalLM.from_pretrained(
            quantized_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        print(f"✓ Quantized model loaded: {type(model_quantized).__name__}")

        # Step 2: Dequantize INT4 weights using wrapper methods
        print("\n[2/3] Dequantizing INT4 weights to FP32...")
        num_layers = model_quantized.config.num_hidden_layers
        for layer_idx in range(num_layers):
            layer = model_quantized.model.layers[layer_idx]

            # Dequantize attention layer
            if hasattr(layer.self_attn, 'dequantize'):
                layer.self_attn.dequantize()
                if layer_idx % 5 == 0:
                    print(f"  ✓ Dequantized layer {layer_idx} attention")

            # Dequantize MLP layer
            if hasattr(layer.mlp, 'dequantize'):
                layer.mlp.dequantize()

        print(f"✓ Dequantized all {num_layers} layers")

        # Step 3: Convert to BF16 and save
        print("\n[3/3] Converting to BF16 for Trainium2...")
        model_quantized.to(torch.bfloat16)
        model_bf16 = model_quantized
        print(f"✓ Model converted to BF16")

        # Step 4: Save BF16 model
        print(f"\nSaving BF16 model to {output_path}")
        Path(output_path).mkdir(parents=True, exist_ok=True)
        model_bf16.save_pretrained(output_path)
        print(f"✓ Model saved successfully")

        # Verify
        print(f"\nVerifying saved model...")
        verify_model = AutoModelForCausalLM.from_pretrained(
            output_path,
            torch_dtype=torch.bfloat16,
            device_map="cpu"
        )
        print(f"✓ Verification passed")

        print("\n" + "=" * 60)
        print("✓ Dequantization Complete!")
        print(f"  Input: {quantized_path} (INT4 quantized)")
        print(f"  Output: {output_path} (BF16 standard)")
        print(f"  Trade-off: Lose INT4 compression, gain Trainium2 compatibility")
        print("=" * 60)

        return True

    except Exception as e:
        print(f"\n✗ Dequantization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Dequantize FlatQuant INT4 model to BF16")
    parser.add_argument(
        "--quantized_model",
        type=str,
        required=True,
        help="Path to quantized model checkpoint from Phase 1"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path for BF16 model"
    )

    args = parser.parse_args()

    success = dequantize_flatquant_model(args.quantized_model, args.output)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
