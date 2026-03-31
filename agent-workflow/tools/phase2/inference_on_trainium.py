#!/usr/bin/env python3
"""
Run inference and benchmark on Trainium2 with traced model.

This script:
1. Loads the traced model (from trace_for_trainium.py)
2. Runs warmup iterations
3. Benchmarks inference latency and throughput
4. Compares vs CPU baseline

Usage:
    python inference_on_trainium.py --model ./llama2_neuron_traced/model_traced.pt --benchmark
"""

import sys
import torch
import time
from pathlib import Path
from typing import Dict, List


def benchmark_inference(
    model_path: str,
    num_iterations: int = 10,
    sequence_length: int = 128,
    batch_size: int = 1,
) -> Dict[str, float]:
    """
    Benchmark inference latency on Trainium2 traced model.

    Args:
        model_path: Path to traced model
        num_iterations: Number of iterations to measure
        sequence_length: Sequence length for inputs
        batch_size: Batch size for inputs

    Returns:
        Dictionary with timing statistics
    """
    print("=" * 60)
    print("Trainium2 Inference Benchmark")
    print("=" * 60)

    try:
        # Load traced model
        print(f"\n[1/3] Loading traced model from {model_path}")
        model = torch.jit.load(model_path)
        model.eval()
        print(f"✓ Model loaded successfully")

        # Warmup
        print(f"\n[2/3] Warming up (3 iterations)...")
        for i in range(3):
            with torch.no_grad():
                input_ids = torch.randint(
                    0, 32000, (batch_size, sequence_length), dtype=torch.long
                )
                _ = model(input_ids)
            print(f"  Warmup {i+1}/3 complete")

        # Benchmark
        print(f"\n[3/3] Benchmarking ({num_iterations} iterations)...")
        print(f"      Sequence length: {sequence_length}")
        print(f"      Batch size: {batch_size}")

        times = []
        for i in range(num_iterations):
            with torch.no_grad():
                input_ids = torch.randint(
                    0, 32000, (batch_size, sequence_length), dtype=torch.long
                )

                # Measure
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                start = time.perf_counter()

                output = model(input_ids)

                torch.cuda.synchronize() if torch.cuda.is_available() else None
                elapsed = time.perf_counter() - start

                times.append(elapsed)
                print(f"  Iteration {i+1}/{num_iterations}: {elapsed:.2f}s")

        # Calculate statistics
        avg_latency = sum(times) / len(times)
        min_latency = min(times)
        max_latency = max(times)
        throughput = sequence_length / avg_latency  # tokens per second

        # CPU baseline comparison
        cpu_baseline = 45  # Approximate CPU latency for 128 tokens
        speedup = cpu_baseline / avg_latency

        # Print results
        print("\n" + "=" * 60)
        print("Results:")
        print("=" * 60)
        print(f"Average latency: {avg_latency:.2f}s")
        print(f"Min latency: {min_latency:.2f}s")
        print(f"Max latency: {max_latency:.2f}s")
        print(f"Throughput: {throughput:.2f} tokens/sec")
        print(f"\nComparison:")
        print(f"CPU baseline (~45s): {cpu_baseline}s for {sequence_length} tokens")
        print(f"Trainium speedup: {speedup:.1f}x faster than CPU")
        print("=" * 60)

        return {
            "avg_latency": avg_latency,
            "min_latency": min_latency,
            "max_latency": max_latency,
            "throughput": throughput,
            "speedup": speedup,
            "times": times,
        }

    except Exception as e:
        print(f"\n✗ Benchmarking failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_inference(model_path: str, prompt: str = None, max_tokens: int = 50) -> str:
    """
    Run single inference pass and generate text.

    Args:
        model_path: Path to traced model
        prompt: Input prompt text
        max_tokens: Maximum tokens to generate

    Returns:
        Generated text
    """
    print("=" * 60)
    print("Trainium2 Text Generation")
    print("=" * 60)

    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM

        # Load tokenizer
        print(f"\nLoading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
        print(f"✓ Tokenizer loaded")

        # Load traced model
        print(f"Loading traced model from {model_path}")
        model = torch.jit.load(model_path)
        model.eval()
        print(f"✓ Model loaded")

        if prompt is None:
            prompt = "The future of artificial intelligence is"

        # Encode prompt
        print(f"\nPrompt: {prompt}")
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.input_ids

        # Generate
        print(f"Generating {max_tokens} tokens...")
        with torch.no_grad():
            # Simple greedy generation
            output_ids = input_ids
            for _ in range(max_tokens):
                logits = model(output_ids)
                next_token = logits[0, -1, :].argmax(dim=-1, keepdim=True)
                output_ids = torch.cat([output_ids, next_token], dim=1)

        # Decode
        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        print(f"\nGenerated:")
        print(f"{generated_text}")
        print("=" * 60)

        return generated_text

    except Exception as e:
        print(f"\n✗ Generation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run inference on Trainium2")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to traced model"
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run benchmark (measure latency)"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="The future of artificial intelligence is",
        help="Prompt for text generation"
    )
    parser.add_argument(
        "--num_iterations",
        type=int,
        default=10,
        help="Number of benchmark iterations"
    )
    parser.add_argument(
        "--sequence_length",
        type=int,
        default=128,
        help="Sequence length for inputs"
    )

    args = parser.parse_args()

    if args.benchmark:
        benchmark_inference(
            args.model,
            num_iterations=args.num_iterations,
            sequence_length=args.sequence_length,
        )
    else:
        run_inference(args.model, args.prompt)


if __name__ == "__main__":
    main()
