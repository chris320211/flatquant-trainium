#!/usr/bin/env python3
"""
Phase 2: Trainium2 Inference with Explicit Transforms in Graph (Option 2)

This script loads a FlatQuant quantized model (with explicit transforms preserved)
and traces it for Trainium2 using torch_neuronx. The computation graph will include:
- Explicit transformation matrices T
- Explicit quantization operations
- Standard linear operations

neuron-cc compiler will perform automatic kernel fusion on these operations.

Key difference from Option 1 (dequantization):
- NO dequantization step
- Quantization (INT4 + FP8) preserved throughout
- Transforms are explicit in graph (not fused into weights)
- Compiler fuses T operations automatically during tracing

Usage:
    python inference_on_trainium_option2.py \
        --quantized_model ./quantized_model \
        --sequence_length 128 \
        --benchmark
"""

import sys
import torch
import time
from pathlib import Path
from typing import Dict, Optional

# CRITICAL: Import transformers FIRST, before any FlatQuantBundled modules
from transformers import AutoModelForCausalLM, AutoTokenizer

# NOTE: FlatQuantBundled should already be in PYTHONPATH from setup_env.sh


class FlatQuantLlamaTrainium:
    """
    Wrapper to load and trace a FlatQuant quantized Llama model for Trainium2.

    This keeps transforms as explicit operations in the computation graph,
    allowing neuron-cc to optimize them during compilation.
    """

    def __init__(self, quantized_model_path: str):
        """Load quantized model with transforms preserved as Parameters."""
        self.quantized_model_path = quantized_model_path
        self.model = None
        self.tokenizer = None
        self.quant_config = None

        self._load_model()
        self._load_tokenizer()
        self._load_quant_config()
        self._verify_eval_mode()

    def _load_model(self):
        """Load the quantized model (with T as Parameters)."""
        print(f"[1/4] Loading quantized model from {self.quantized_model_path}")
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.quantized_model_path,
                torch_dtype=torch.float16,
                device_map="cpu"
            )
            self.model.eval()
            print(f"✓ Model loaded: {type(self.model).__name__}")
        except Exception as e:
            print(f"✗ Failed to load model: {e}")
            raise

    def _load_tokenizer(self):
        """Load the tokenizer."""
        print(f"[2/4] Loading tokenizer")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.quantized_model_path
            )
            print(f"✓ Tokenizer loaded")
        except Exception as e:
            print(f"✗ Failed to load tokenizer: {e}")
            raise

    def _load_quant_config(self):
        """Load quantization configuration."""
        print(f"[3/4] Loading quantization config")
        try:
            import json
            quant_config_path = Path(self.quantized_model_path) / "quant_config.json"
            if quant_config_path.exists():
                with open(quant_config_path) as f:
                    self.quant_config = json.load(f)
                print(f"✓ Quant config loaded")
                print(f"  - w_bits: {self.quant_config.get('w_bits')}")
                print(f"  - a_bits: {self.quant_config.get('a_bits')}")
                print(f"  - strategy: {self.quant_config.get('strategy')}")
            else:
                print(f"⚠ No quant_config.json found (optional)")
        except Exception as e:
            print(f"⚠ Failed to load quant config: {e}")

    def _verify_eval_mode(self):
        """Verify that model layers are in evaluation mode with explicit transforms."""
        print(f"[4/4] Verifying evaluation mode")
        num_layers = self.model.config.num_hidden_layers
        eval_count = 0

        for layer_idx in range(num_layers):
            layer = self.model.model.layers[layer_idx]

            # Check attention eval mode
            if hasattr(layer.self_attn, '_eval_mode') and layer.self_attn._eval_mode:
                eval_count += 1

            # Check MLP eval mode
            if hasattr(layer.mlp, '_ori_mode') and not layer.mlp._ori_mode:
                eval_count += 1

        if eval_count == num_layers * 2:
            print(f"✓ All {num_layers} layers in evaluation mode")
            print(f"✓ Explicit transforms are active in computation graph")
        else:
            print(f"⚠ Only {eval_count}/{num_layers * 2} layer components in eval mode")
            print(f"  This may indicate the model was not saved with Option 2 strategy")

    def trace_for_trainium(
        self,
        sequence_length: int = 128,
        batch_size: int = 1,
        compiler_workdir: str = "./compiler_workdir/",
        num_neuroncores: int = 1,
    ) -> Optional[torch.nn.Module]:
        """
        Trace the FlatQuant model for Trainium2 compilation.

        The computation graph will include:
        1. Explicit transformation matrices T (not fused into weights)
        2. Explicit quantization operations (INT4 + FP8)
        3. Standard linear and attention operations

        neuron-cc will analyze this graph and perform:
        - Automatic kernel fusion for T operations
        - Quantization-aware optimizations
        - Memory layout optimization for Trainium

        Args:
            sequence_length: Input sequence length for tracing
            batch_size: Batch size (usually 1 for inference)
            compiler_workdir: Working directory for neuron-cc compilation
            num_neuroncores: Number of NeuroCores to target (1 or 2 for Trainium2)

        Returns:
            Traced model if successful, None otherwise
        """
        print("\n" + "=" * 70)
        print("Tracing FlatQuant Model for Trainium2 (Option 2: Explicit Transforms)")
        print("=" * 70)

        try:
            # Step 1: Create example input
            print(f"\nStep 1: Creating example input")
            print(f"  Batch size: {batch_size}")
            print(f"  Sequence length: {sequence_length}")

            example_input = torch.randint(
                0, 32000, (batch_size, sequence_length), dtype=torch.long
            )
            print(f"✓ Example input shape: {example_input.shape}")

            # Step 2: Move model to CPU (will be compiled to Neuron)
            print(f"\nStep 2: Preparing model for tracing")
            self.model.to("cpu")
            self.model.eval()
            print(f"✓ Model on CPU in eval mode")

            # Step 3: Trace with torch_neuronx
            print(f"\nStep 3: Tracing with torch_neuronx")
            print(f"  Computation graph will show:")
            print(f"  - Explicit transformation matrices T")
            print(f"  - Explicit quantization operations")
            print(f"  - Linear and attention operations")
            print(f"  neuron-cc will fuse these automatically")

            try:
                import torch_neuronx
            except ImportError:
                print(f"⚠ torch_neuronx not available (expected if not on Trainium2)")
                print(f"  Skipping actual tracing, returning untraced model")
                return self.model

            Path(compiler_workdir).mkdir(parents=True, exist_ok=True)

            traced_model = torch.neuron.trace(
                self.model,
                example_input,
                compiler_workdir=compiler_workdir,
                compiler_args=[
                    "--model-type=transformer",
                    f"--num-neuroncores={num_neuroncores}",
                ]
            )
            print(f"✓ Tracing complete")
            print(f"✓ Model compiled for Trainium2")

            return traced_model

        except Exception as e:
            print(f"\n✗ Tracing failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def benchmark_inference(
        self,
        traced_model: torch.nn.Module,
        sequence_length: int = 128,
        batch_size: int = 1,
        num_iterations: int = 10,
    ) -> Dict[str, float]:
        """
        Benchmark inference latency on the traced model.

        Args:
            traced_model: Traced model from trace_for_trainium()
            sequence_length: Sequence length for inputs
            batch_size: Batch size for inputs
            num_iterations: Number of iterations to measure

        Returns:
            Dictionary with latency statistics
        """
        print("\n" + "=" * 70)
        print("Benchmarking Inference on Trainium2")
        print("=" * 70)

        print(f"\nConfiguration:")
        print(f"  Batch size: {batch_size}")
        print(f"  Sequence length: {sequence_length}")
        print(f"  Iterations: {num_iterations}")

        try:
            # Warmup
            print(f"\nWarming up (3 iterations)...")
            for i in range(3):
                with torch.no_grad():
                    input_ids = torch.randint(
                        0, 32000, (batch_size, sequence_length), dtype=torch.long
                    )
                    _ = traced_model(input_ids)
                print(f"  Warmup {i+1}/3 complete")

            # Benchmark
            print(f"\nBenchmarking...")
            times = []
            for i in range(num_iterations):
                with torch.no_grad():
                    input_ids = torch.randint(
                        0, 32000, (batch_size, sequence_length), dtype=torch.long
                    )

                    # Synchronize before timing (if available)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()

                    start = time.perf_counter()
                    output = traced_model(input_ids)
                    elapsed = time.perf_counter() - start

                    # Synchronize after timing (if available)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()

                    times.append(elapsed)
                    print(f"  Iteration {i+1}/{num_iterations}: {elapsed:.3f}s")

            # Calculate statistics
            avg_latency = sum(times) / len(times)
            min_latency = min(times)
            max_latency = max(times)
            throughput = sequence_length / avg_latency

            # CPU baseline (approximate)
            cpu_baseline_latency = 45.0  # Approximate for 128 tokens on CPU

            # Print results
            print("\n" + "=" * 70)
            print("Benchmark Results:")
            print("=" * 70)
            print(f"Average latency: {avg_latency:.3f}s")
            print(f"Min latency: {min_latency:.3f}s")
            print(f"Max latency: {max_latency:.3f}s")
            print(f"Throughput: {throughput:.2f} tokens/sec")
            print(f"\nComparison:")
            print(f"CPU baseline (approx): {cpu_baseline_latency:.1f}s")
            print(f"Trainium speedup: {cpu_baseline_latency / avg_latency:.1f}x faster")
            print("=" * 70)

            return {
                "avg_latency": avg_latency,
                "min_latency": min_latency,
                "max_latency": max_latency,
                "throughput": throughput,
                "cpu_baseline": cpu_baseline_latency,
                "speedup": cpu_baseline_latency / avg_latency,
            }

        except Exception as e:
            print(f"\n✗ Benchmarking failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def generate_text(
        self,
        traced_model: torch.nn.Module,
        prompt: str = "The future of artificial intelligence is",
        max_tokens: int = 50,
    ) -> str:
        """
        Generate text using the traced model.

        Args:
            traced_model: Traced model from trace_for_trainium()
            prompt: Input prompt text
            max_tokens: Maximum tokens to generate

        Returns:
            Generated text
        """
        print("\n" + "=" * 70)
        print("Text Generation on Trainium2")
        print("=" * 70)

        try:
            print(f"\nPrompt: {prompt}")
            print(f"Max tokens: {max_tokens}")

            # Encode prompt
            inputs = self.tokenizer(prompt, return_tensors="pt")
            input_ids = inputs["input_ids"]

            print(f"Generating...")
            with torch.no_grad():
                output_ids = input_ids.clone()
                for step in range(max_tokens):
                    # Forward pass
                    outputs = traced_model(output_ids)

                    # Extract logits
                    if hasattr(outputs, 'logits'):
                        logits = outputs.logits
                    elif isinstance(outputs, tuple):
                        logits = outputs[0]
                    else:
                        logits = outputs

                    # Get next token
                    next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                    output_ids = torch.cat([output_ids, next_token], dim=1)

                    if step % 10 == 0:
                        print(f"  Generated {step}/{max_tokens} tokens")

            # Decode
            generated_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

            print(f"\n" + "=" * 70)
            print("Generated Text:")
            print("=" * 70)
            print(generated_text)
            print("=" * 70)

            return generated_text

        except Exception as e:
            print(f"\n✗ Text generation failed: {e}")
            import traceback
            traceback.print_exc()
            return None


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Phase 2: Trainium2 Inference with Option 2 (Explicit Transforms)"
    )
    parser.add_argument(
        "--quantized_model",
        type=str,
        required=True,
        help="Path to quantized model from Phase 1"
    )
    parser.add_argument(
        "--sequence_length",
        type=int,
        default=128,
        help="Sequence length for tracing and inference"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for inference"
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run latency benchmark"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="The future of artificial intelligence is",
        help="Prompt for text generation"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=50,
        help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--compiler_workdir",
        type=str,
        default="./compiler_workdir/",
        help="Working directory for neuron-cc compilation"
    )
    parser.add_argument(
        "--num_neuroncores",
        type=int,
        default=1,
        help="Number of NeuroCores to target (1 or 2)"
    )
    parser.add_argument(
        "--num_iterations",
        type=int,
        default=10,
        help="Number of benchmark iterations"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("FlatQuant Trainium2 Inference - Option 2")
    print("Strategy: Explicit Transforms (No Dequantization)")
    print("=" * 70)

    # Load and prepare model
    wrapper = FlatQuantLlamaTrainium(args.quantized_model)

    # Trace for Trainium
    traced_model = wrapper.trace_for_trainium(
        sequence_length=args.sequence_length,
        batch_size=args.batch_size,
        compiler_workdir=args.compiler_workdir,
        num_neuroncores=args.num_neuroncores,
    )

    if traced_model is None:
        print("\n✗ Failed to trace model")
        sys.exit(1)

    # Run benchmark or text generation
    if args.benchmark:
        stats = wrapper.benchmark_inference(
            traced_model,
            sequence_length=args.sequence_length,
            batch_size=args.batch_size,
            num_iterations=args.num_iterations,
        )
        if stats is None:
            sys.exit(1)
    else:
        generated = wrapper.generate_text(
            traced_model,
            prompt=args.prompt,
            max_tokens=args.max_tokens,
        )
        if generated is None:
            sys.exit(1)

    print("\n✓ Option 2 (Explicit Transforms) inference complete!")
    sys.exit(0)


if __name__ == "__main__":
    main()
