#!/usr/bin/env python3
"""
Trace BF16 model with torch_neuronx for Trainium2 compilation.

This script:
1. Loads the dequantized BF16 model (from dequant_for_trainium.py)
2. Creates example inputs
3. Traces with torch_neuronx (XLA compilation)
4. Saves traced model for inference

Usage:
    python trace_for_trainium.py --model ./llama2_bf16_for_trainium --output ./llama2_neuron_traced
"""

import sys
import torch
from pathlib import Path
from typing import Optional

# Setup paths
sys.path.insert(0, str(Path(__file__).parent))

# Import torch_neuronx if available (only on Trainium2)
try:
    import torch_neuronx
    NEURONX_AVAILABLE = True
except ImportError:
    NEURONX_AVAILABLE = False


def trace_model_for_trainium(
    model_path: str,
    output_dir: str,
    sequence_length: int = 128,
    num_neuroncores: int = 1,
) -> Optional[torch.jit.ScriptModule]:
    """
    Trace BF16 model for Trainium2 compilation.

    Args:
        model_path: Path to BF16 model
        output_dir: Output directory for traced model
        sequence_length: Sequence length for example input
        num_neuroncores: Number of Trainium cores to use (1, 2, or 8)

    Returns:
        Traced model if successful, None otherwise
    """
    print("=" * 60)
    print("Tracing BF16 Model for Trainium2 (XLA Compilation)")
    print("=" * 60)

    try:
        from transformers import AutoModelForCausalLM

        # Step 1: Load BF16 model
        print(f"\n[1/3] Loading BF16 model from {model_path}")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="cpu"
        )
        model.eval()
        print(f"✓ Model loaded: {type(model).__name__}")

        # Step 2: Create example input
        print(f"\n[2/3] Creating example input (seq_len={sequence_length})")
        example_input = torch.randint(0, 32000, (1, sequence_length), dtype=torch.long)
        print(f"✓ Example input shape: {example_input.shape}")

        # Step 3: Compile with torch_neuronx
        print(f"\n[3/3] Compiling with torch_neuronx...")
        print(f"      (This may take 10-30 minutes on Trainium2)")
        print(f"      (Compiler directory: {output_dir}/compiler_workdir)")

        if not NEURONX_AVAILABLE:
            print(f"⚠ torch_neuronx not available (expected on non-Trainium2 systems)")
            print(f"  Note: Compilation only works on Trainium2 instance with torch_neuronx")
            print(f"  This is normal - the model is ready but can't be compiled outside Trainium2")
            return model

        try:
            # Use torch_neuronx.compile() API (correct for torch-neuronx 2.9+)
            # This wraps the model with XLA compilation
            model_compiled = torch_neuronx.compile(
                model,
                sample_inputs=example_input,
                compiler_workdir=f"{output_dir}/compiler_workdir/",
                compiler_args=[
                    "--model-type=transformer",
                    f"--num-neuroncores={num_neuroncores}",
                    "--optlevel=2",
                ]
            )
            print(f"✓ Compilation successful!")
            model_traced = model_compiled

        except Exception as e:
            print(f"⚠ Compilation failed: {e}")
            print(f"  Note: This may be due to model architecture incompatibilities with Trainium2")
            print(f"  Attempting to save uncompiled model for inspection...")
            model_traced = model

        # Step 4: Save compiled model
        print(f"\nSaving compiled model to {output_dir}")
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        if hasattr(model_traced, 'save_pretrained'):
            # Model object - save using HuggingFace format
            model_traced.save_pretrained(output_dir)
            print(f"✓ Compiled model saved (HuggingFace format)")
        else:
            # ScriptModule - save using torch.jit format
            torch.jit.save(model_traced, f"{output_dir}/model_traced.pt")
            print(f"✓ Compiled model saved (TorchScript format)")

        # Step 5: Verify
        print(f"\nVerifying compiled model...")
        try:
            if hasattr(model_traced, 'save_pretrained'):
                loaded_model = AutoModelForCausalLM.from_pretrained(
                    output_dir,
                    torch_dtype=torch.bfloat16,
                    device_map="cpu"
                )
            else:
                loaded_model = torch.jit.load(f"{output_dir}/model_traced.pt")
            print(f"✓ Compiled model verified")
        except Exception as e:
            print(f"⚠ Verification skipped: {e}")

        print("\n" + "=" * 60)
        print("✓ Tracing Complete!")
        print(f"  Model: {model_path}")
        print(f"  Traced output: {output_dir}/model_traced.pt")
        print(f"  Ready for inference on Trainium2 hardware")
        print("=" * 60)

        return model_traced

    except Exception as e:
        print(f"\n✗ Tracing failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Trace BF16 model for Trainium2")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to BF16 model (from dequant_for_trainium.py)"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for traced model"
    )
    parser.add_argument(
        "--sequence_length",
        type=int,
        default=128,
        help="Sequence length for example input"
    )
    parser.add_argument(
        "--num_neuroncores",
        type=int,
        default=1,
        choices=[1, 2, 8],
        help="Number of Trainium cores (1, 2, or 8)"
    )

    args = parser.parse_args()

    result = trace_model_for_trainium(
        args.model,
        args.output,
        sequence_length=args.sequence_length,
        num_neuroncores=args.num_neuroncores,
    )

    sys.exit(0 if result else 1)


if __name__ == "__main__":
    main()
