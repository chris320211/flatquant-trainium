#!/usr/bin/env python3
"""
Verify that Phase 1 quantized model produces coherent text.

This tests inference directly on the quantized checkpoint WITHOUT dequantization,
to see if the problem is in Phase 1 or Phase 2.

Usage:
    python verify_quantized.py --model ./quantized_llama2_7b --prompt "The future is"
"""

import sys
import torch
from pathlib import Path

# Add FlatQuant to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "FlatQuantBundled"))

from transformers import AutoModelForCausalLM, AutoTokenizer

def verify_quantized_inference(model_path: str, prompt: str = "The future of AI is"):
    """
    Test inference on quantized model to verify Phase 1 works.

    Args:
        model_path: Path to quantized checkpoint
        prompt: Text prompt for generation
    """
    print("=" * 60)
    print("Verify Phase 1 Quantized Model Inference")
    print("=" * 60)

    try:
        # Load model and tokenizer
        print(f"\n[1/3] Loading quantized model from {model_path}")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            local_files_only=True
        )
        model.eval()
        print(f"✓ Model loaded: {type(model).__name__}")

        print(f"\n[2/3] Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True
        )
        print(f"✓ Tokenizer loaded")

        # Test inference
        print(f"\n[3/3] Running inference on quantized model...")
        print(f"Prompt: {prompt}")
        print(f"Generating 30 tokens...")

        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(model.device)

        with torch.no_grad():
            output_ids = input_ids.clone()
            for i in range(30):
                outputs = model(output_ids)
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
                next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                output_ids = torch.cat([output_ids, next_token], dim=1)
                if i % 10 == 0:
                    print(f"  Generated {i}/30 tokens")

        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        print("\n" + "=" * 60)
        print("Generated text:")
        print(generated_text)
        print("=" * 60)

        # Quick analysis
        tokens_list = tokenizer.convert_ids_to_tokens(output_ids[0])
        unique_tokens = len(set(tokens_list))

        print(f"\nAnalysis:")
        print(f"  Total tokens: {len(tokens_list)}")
        print(f"  Unique tokens: {unique_tokens}")
        print(f"  Token diversity: {unique_tokens / len(tokens_list) * 100:.1f}%")

        # Check if output looks like garbage
        if unique_tokens < 5:
            print(f"\n  ⚠️  LOW DIVERSITY - Output looks like repeated garbage!")
            return False
        else:
            print(f"\n  ✓ Output has reasonable diversity")
            return True

    except Exception as e:
        print(f"\n✗ Verification failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Verify quantized model inference")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to quantized model checkpoint"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="The future of artificial intelligence is",
        help="Text prompt for generation"
    )

    args = parser.parse_args()

    success = verify_quantized_inference(args.model, args.prompt)
    sys.exit(0 if success else 1)
