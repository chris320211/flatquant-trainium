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

    Key insight: The saved "quantized" checkpoint is actually the REPARAMETERIZED model
    with FP16 weights (not INT4 bytes). The reparameterization already happened during
    Phase 1 save_model(). So we just need to:
    1. Load the reparameterized FP16 weights
    2. Convert to BF16 for Trainium2 compatibility
    3. Save

    Args:
        quantized_path: Path to reparameterized model checkpoint (from Phase 1)
        output_path: Path to save BF16 model

    Returns:
        True if successful, False otherwise
    """
    print("=" * 60)
    print("FlatQuant Reparameterized → BF16 Conversion for Trainium2")
    print("=" * 60)

    try:
        # Step 1: Load reparameterized model (stored as FP16)
        print(f"\n[1/2] Loading reparameterized model from {quantized_path}")
        model_fp16 = AutoModelForCausalLM.from_pretrained(
            quantized_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        print(f"✓ Model loaded: {type(model_fp16).__name__}")

        # Verify weights are reasonable (not garbage quantized values)
        sample_weight = model_fp16.model.layers[0].self_attn.q_proj.weight
        print(f"  Sample weight range: [{sample_weight.min():.4f}, {sample_weight.max():.4f}]")

        # Step 2: Convert to BF16 for Trainium2
        print(f"\n[2/2] Converting to BF16 for Trainium2...")
        model_bf16 = model_fp16.to(torch.bfloat16)
        print(f"✓ Model converted to BF16")

        # Step 3: Save BF16 model
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
        print("✓ Conversion Complete!")
        print(f"  Input: {quantized_path} (FP16 reparameterized)")
        print(f"  Output: {output_path} (BF16 standard)")
        print("=" * 60)

        return True

    except Exception as e:
        print(f"\n✗ Conversion failed: {e}")
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
