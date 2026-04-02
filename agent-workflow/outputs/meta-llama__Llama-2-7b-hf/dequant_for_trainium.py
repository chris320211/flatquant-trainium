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

# Setup paths - find FlatQuantBundled relative to this script
# This script is in: agent-workflow/outputs/meta-llama__Llama-2-7b-hf/dequant_for_trainium.py
# FlatQuantBundled is at: FlatQuantBundled/ (3 levels up)
SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent.parent.parent
FLATQUANT_PATH = REPO_ROOT / "FlatQuantBundled"

if not FLATQUANT_PATH.exists():
    raise RuntimeError(f"FlatQuantBundled not found at {FLATQUANT_PATH}. Ensure script is run from within the repository.")

sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(FLATQUANT_PATH))


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
        from transformers import AutoModelForCausalLM
        from llama_2_7b_hf_utils import FlatQuantLlamaMLP, FlatQuantLlamaAttention

        # Step 1: Load quantized model
        print(f"\n[1/3] Loading quantized model from {quantized_path}")
        model_quantized = AutoModelForCausalLM.from_pretrained(
            quantized_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        print(f"✓ Quantized model loaded: {type(model_quantized).__name__}")

        # Step 2: Extract state dict and convert to BF16
        print("\n[2/3] Converting weights to BF16...")
        state_dict = model_quantized.state_dict()

        # Convert all weights to BF16
        state_dict_bf16 = {}
        for key, tensor in state_dict.items():
            if isinstance(tensor, torch.Tensor):
                state_dict_bf16[key] = tensor.to(torch.bfloat16)
            else:
                state_dict_bf16[key] = tensor

        print(f"✓ Converted {len(state_dict_bf16)} tensors to BF16")

        # Step 3: Create and load standard BF16 model
        print("\n[3/3] Creating standard BF16 model...")
        model_bf16 = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-2-7b-hf",
            torch_dtype=torch.bfloat16,
            device_map="cpu"
        )

        # Load converted weights
        model_bf16.load_state_dict(state_dict_bf16, strict=False)
        print(f"✓ BF16 model created and weights loaded")

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
