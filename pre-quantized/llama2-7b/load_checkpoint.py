#!/usr/bin/env python3
"""
Load FlatQuant W4A4KV4 checkpoint for LLaMA-2-7B and validate it works.

Prerequisites:
    pip install -r requirements.txt

This script requires FlatQuant to be installed. If not installed, run:
    pip install torch transformers
    pip install git+https://github.com/ruikangliu/FlatQuant.git
"""
import argparse
import sys

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from flatquant.flat_linear import FlatQuantLlamaForCausalLM
except ImportError as e:
    print(f"Error: Missing required dependency: {e}")
    print("\nPlease install dependencies first:")
    print("  pip install -r requirements.txt")
    print("\nOr manually:")
    print("  pip install torch transformers")
    print("  pip install git+https://github.com/ruikangliu/FlatQuant.git")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Load FlatQuant checkpoint")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to base LLaMA-2-7B model")
    parser.add_argument("--matrix_path", type=str, required=True,
                        help="Path to FlatQuant transformation matrices")
    args = parser.parse_args()

    print(f"\n{'='*80}")
    print("Loading LLaMA-2-7B with FlatQuant W4A4KV4")
    print(f"{'='*80}")
    print(f"Model path: {args.model_path}")
    print(f"Matrix path: {args.matrix_path}")
    print(f"{'='*80}\n")

    # Load base model in FP16
    print("Loading base model in FP16...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True
    )
    print(f"Base model loaded: {type(model).__name__}")

    # Apply FlatQuant transformation
    print(f"\nApplying FlatQuant transformation from {args.matrix_path}...")
    model = FlatQuantLlamaForCausalLM(
        model,
        wbits=4,
        abits=4,
        kvbits=4,
        group_size=128,
        reload_matrix=True,
        matrix_path=args.matrix_path
    )
    print("FlatQuant transformation applied")

    # Print model architecture
    print(f"\n{'='*80}")
    print("Model Architecture")
    print(f"{'='*80}")
    print(model)

    # Check quantized layers
    print(f"\n{'='*80}")
    print("Checking Quantized Layers")
    print(f"{'='*80}")
    quantized_layers = []
    for name, module in model.named_modules():
        if "FlatQuant" in type(module).__name__ or hasattr(module, 'wbits'):
            quantized_layers.append(name)

    print(f"Found {len(quantized_layers)} quantized layers")
    if quantized_layers:
        print("Sample quantized layers:")
        for layer in quantized_layers[:5]:
            print(f"  - {layer}")
        if len(quantized_layers) > 5:
            print(f"  ... and {len(quantized_layers) - 5} more")

    # Check weight dtype
    print(f"\n{'='*80}")
    print("Weight Dtype Check")
    print(f"{'='*80}")
    sample_weight = None
    sample_name = None
    for name, param in model.named_parameters():
        if 'weight' in name:
            sample_weight = param
            sample_name = name
            break

    if sample_weight is not None:
        print(f"Sample weight tensor: {sample_name}")
        print(f"  Shape: {sample_weight.shape}")
        print(f"  Dtype: {sample_weight.dtype}")
        print(f"  Device: {sample_weight.device}")

    # Load tokenizer for dummy forward pass
    print(f"\n{'='*80}")
    print("Running Dummy Forward Pass")
    print(f"{'='*80}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.pad_token = tokenizer.eos_token

    # Tokenize a short input
    test_input = "Hello, I am a language model"
    print(f"Input text: '{test_input}'")
    inputs = tokenizer(test_input, return_tensors="pt").to(model.device)
    print(f"Input IDs shape: {inputs['input_ids'].shape}")

    # Run forward pass
    print("Running forward pass...")
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)

    print(f"Output logits shape: {outputs.logits.shape}")
    print(f"Output logits dtype: {outputs.logits.dtype}")

    # Decode output
    next_token_logits = outputs.logits[0, -1, :]
    next_token_id = torch.argmax(next_token_logits).item()
    next_token = tokenizer.decode([next_token_id])
    print(f"Next predicted token: '{next_token}'")

    print(f"\n{'='*80}")
    print("SUCCESS: Model loaded and forward pass completed!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
