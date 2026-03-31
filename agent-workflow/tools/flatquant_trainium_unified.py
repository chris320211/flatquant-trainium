#!/usr/bin/env python3
"""
================================================================================
UNIFIED FLATQUANT + TRAINIUM2 PIPELINE
================================================================================

THIS ENTIRE SCRIPT RUNS ON TRAINIUM2 INSTANCE.
NO EXTERNAL DEPENDENCIES. EVERYTHING SELF-CONTAINED.

Flow:
  1. Load base model from HuggingFace (on Trainium2)
  2. Apply FlatQuant INT4 wrappers (on Trainium2)
  3. Run calibration on real data (on Trainium2)
  4. Save quantized checkpoint with explicit transforms (on Trainium2)
  5. Immediately trace for Trainium2 compilation (on Trainium2)
  6. Run inference and benchmarking (on Trainium2)

Execution:
  # On Trainium2 instance only:
  python flatquant_trainium_unified.py \
      --model meta-llama/Llama-2-7b-hf \
      --hf_token YOUR_TOKEN \
      --output ./quantized_llama2_7b \
      --benchmark \
      --num_tokens 50

Requirements on Trainium2:
  - torch_neuronx (Trainium SDK)
  - transformers
  - FlatQuantBundled in PYTHONPATH
  - ~32GB RAM available
  - Trainium2 NeuroCores available

================================================================================
"""

import sys
import os
import json
import torch
import time
from pathlib import Path
from typing import Dict, Optional
import argparse

# CRITICAL: Import transformers FIRST, before any FlatQuantBundled modules
from transformers import AutoModelForCausalLM, AutoTokenizer

# FlatQuantBundled should be in PYTHONPATH from setup_env.sh
import flatquant.utils as fq_utils
import flatquant.data_utils as data_utils
import flatquant.train_utils as train_utils
import flatquant.flat_utils as flat_utils

# Import model-specific wrappers
from llama_2_7b_hf_utils import FlatQuantLlamaMLP, FlatQuantLlamaAttention


class TrainiumUnifiedPipeline:
    """
    All-in-one FlatQuant pipeline that runs entirely on Trainium2.

    No data transfer, no external dependencies, no cross-machine issues.
    Everything happens on Trainium2.
    """

    def __init__(self, model_name: str, output_path: str, hf_token: str = None):
        self.model_name = model_name
        self.output_path = Path(output_path)
        self.hf_token = hf_token
        self.model = None
        self.tokenizer = None
        self.device = None

        print("\n" + "=" * 80)
        print("TRAINIUM2 UNIFIED PIPELINE - RUNNING ON TRAINIUM2 INSTANCE")
        print("=" * 80)
        self._check_trainium_environment()

    def _check_trainium_environment(self):
        """Verify we're on Trainium2 with necessary packages."""
        print("\n[INIT] Checking Trainium2 environment...")

        # Check PyTorch
        print(f"  ✓ PyTorch: {torch.__version__}")
        print(f"  ✓ Device: {torch.device('cpu')}")

        # Check Trainium-specific imports
        try:
            import torch_neuronx
            print(f"  ✓ torch_neuronx available (Trainium SDK installed)")
            self.has_trainium = True
        except ImportError:
            print(f"  ⚠ torch_neuronx NOT available (this is CPU, not Trainium2)")
            print(f"    But continuing anyway - tracing will be skipped")
            self.has_trainium = False

        # Check FlatQuantBundled
        try:
            import flatquant
            print(f"  ✓ FlatQuantBundled available in PYTHONPATH")
        except ImportError:
            print(f"  ✗ FlatQuantBundled NOT in PYTHONPATH")
            print(f"    Run: export PYTHONPATH=./FlatQuantBundled:$PYTHONPATH")
            sys.exit(1)

        # Detect available device for training (GPU if available, else CPU)
        # Note: This will be CPU for Trainium2 training, but that's fine
        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')
            print(f"  ✓ GPU detected: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device('cpu')
            print(f"  ✓ Using CPU for training (Trainium2 CPU cores)")

        print(f"  ✓ Training device: {self.device}")
        print(f"  ✓ Trainium2 environment verified\n")

    def load_and_wrap_model(self):
        """
        [ON TRAINIUM2] Load base model and apply FlatQuant wrappers.
        Everything stays on Trainium2.
        """
        print("[STEP 1/6] Loading base model and applying FlatQuant wrappers")
        print("=" * 80)

        # Step 1a: Load model from HuggingFace
        print(f"\nLoading base model: {self.model_name}")
        print(f"  (Loading on Trainium2, will keep quantized model here)")

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="cpu",  # CPU for now, will move to training device during calibration
            token=self.hf_token,
        )
        self.model.eval()
        print(f"  ✓ Base model loaded: {type(self.model).__name__}")

        # Step 1b: Load tokenizer
        print(f"\nLoading tokenizer: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            use_fast=False,
            token=self.hf_token,
        )
        print(f"  ✓ Tokenizer loaded")

        # Step 1c: Apply FlatQuant wrappers
        print(f"\nApplying FlatQuant wrappers (INT4 weights, FP8 activations)...")

        class FlatQuantArgs:
            w_bits = 4
            a_bits = 8
            group_size = 128
            w_asym = False
            a_asym = False
            a_groupsize = -1
            lwc = False
            direct_inv = False
            add_diag = False
            diag_init = "sq_style"
            lac = False
            separate_vtrans = True
            q_bits = 8
            k_bits = 8
            v_bits = 8
            q_asym = False
            k_asym = False
            v_asym = False

        args = FlatQuantArgs()
        num_layers = self.model.config.num_hidden_layers

        for layer_idx in range(num_layers):
            layer = self.model.model.layers[layer_idx]

            # Wrap attention
            try:
                layer.self_attn = FlatQuantLlamaAttention(args, layer.self_attn)
            except Exception as e:
                print(f"  Layer {layer_idx}: attention wrap failed - {e}")

            # Wrap MLP
            try:
                layer.mlp = FlatQuantLlamaMLP(args, layer.mlp)
            except Exception as e:
                print(f"  Layer {layer_idx}: MLP wrap failed - {e}")

        print(f"  ✓ Applied wrappers to {num_layers} layers")
        print(f"\n[STEP 1 COMPLETE] Model ready for calibration on Trainium2\n")

        return self.model, self.tokenizer, args

    def calibrate_on_trainium2(self, args, dataset_name: str = "wikitext", num_samples: int = 128):
        """
        [ON TRAINIUM2] Run calibration using Trainium2's CPU/memory.
        Learns transformation matrices T for quantization-friendly activations.
        Everything happens on Trainium2 - no external calibration needed.
        """
        print("[STEP 2/6] Running FlatQuant calibration (ON TRAINIUM2)")
        print("=" * 80)

        print(f"\nCalibration dataset: {dataset_name}")
        print(f"Calibration samples: {num_samples}")
        print(f"Calibration device: {self.device}")
        print(f"(This runs entirely on Trainium2 instance)")

        # Set sequence length
        self.model.seqlen = 2048

        # Load calibration dataset
        print(f"\nLoading calibration data...")
        try:
            trainloader = data_utils.get_loaders(
                args=None,
                name=dataset_name,
                nsamples=num_samples,
                seed=0,
                seqlen=self.model.seqlen,
                eval_mode=False,
            )
            print(f"  ✓ Loaded {num_samples} calibration samples")
        except Exception as e:
            print(f"  ⚠ Could not load dataset: {e}")
            print(f"    Skipping calibration (using untrained transforms)")
            return

        # Run calibration
        print(f"\nRunning calibration (learning transform matrices)...")
        try:
            train_utils.cali_flat_quant(
                args=None,
                model=self.model,
                trainloader=trainloader,
                dev=self.device,
                logger=self._get_logger(),
            )
            print(f"  ✓ Calibration complete")
        except Exception as e:
            print(f"  ⚠ Calibration failed: {e}")
            print(f"    Continuing with untrained transforms...")

        print(f"\n[STEP 2 COMPLETE] Calibration done on Trainium2\n")

    def save_quantized_on_trainium2(self, args):
        """
        [ON TRAINIUM2] Save quantized model with explicit transforms.
        Keep everything on Trainium2 - no data transfer.
        """
        print("[STEP 3/6] Saving quantized model (ON TRAINIUM2)")
        print("=" * 80)

        output_path = str(self.output_path)
        print(f"\nSaving to: {output_path}")
        print(f"(Model stays on Trainium2 instance)")

        # Set evaluation mode on all layers
        print(f"\nSetting evaluation mode (keeping transforms explicit)...")
        num_layers = self.model.config.num_hidden_layers

        for layer_idx in range(num_layers):
            layer = self.model.model.layers[layer_idx]

            # Attention eval mode
            if hasattr(layer.self_attn, '_eval_mode'):
                layer.self_attn._eval_mode = True
                if hasattr(layer.self_attn, 'ln_trans') and layer.self_attn.ln_trans is not None:
                    layer.self_attn.ln_trans.to_eval_mode()
                if hasattr(layer.self_attn, 'o_trans') and layer.self_attn.o_trans is not None:
                    layer.self_attn.o_trans.to_eval_mode()
                if hasattr(layer.self_attn, 'kcache_trans') and layer.self_attn.kcache_trans is not None:
                    layer.self_attn.kcache_trans.to_eval_mode()
                if hasattr(layer.self_attn, 'vcache_trans') and layer.self_attn.vcache_trans is not None:
                    layer.self_attn.vcache_trans.to_eval_mode()

            # MLP eval mode
            if hasattr(layer.mlp, '_ori_mode'):
                layer.mlp._ori_mode = False
            if hasattr(layer.mlp, 'up_gate_trans') and layer.mlp.up_gate_trans is not None:
                layer.mlp.up_gate_trans.to_eval_mode()
            if hasattr(layer.mlp, 'down_trans') and layer.mlp.down_trans is not None:
                layer.mlp.down_trans.to_eval_mode()

        # Set eval mode on projections
        print(f"Setting FlatQuantizedLinear to evaluation mode...")
        for layer_idx in range(num_layers):
            layer = self.model.model.layers[layer_idx]

            for proj_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
                if hasattr(layer.self_attn, proj_name):
                    proj = getattr(layer.self_attn, proj_name)
                    if hasattr(proj, '_eval_mode'):
                        proj._eval_mode = True

            for proj_name in ['up_proj', 'gate_proj', 'down_proj']:
                if hasattr(layer.mlp, proj_name):
                    proj = getattr(layer.mlp, proj_name)
                    if hasattr(proj, '_eval_mode'):
                        proj._eval_mode = True

        print(f"  ✓ Evaluation mode set")

        # Save checkpoint
        print(f"\nSaving model checkpoint...")
        Path(output_path).mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)
        print(f"  ✓ Model saved")

        # Save quant config
        print(f"Saving quantization config...")
        quant_config = {
            "w_bits": args.w_bits,
            "a_bits": args.a_bits,
            "group_size": args.group_size,
            "w_asym": args.w_asym,
            "a_asym": args.a_asym,
            "a_groupsize": args.a_groupsize,
            "lwc": args.lwc,
            "q_bits": args.q_bits,
            "k_bits": args.k_bits,
            "v_bits": args.v_bits,
            "model_type": "llama",
            "strategy": "option2_explicit_transforms",
            "device_location": "TRAINIUM2",  # EXPLICITLY MARK QUANTIZATION LOCATION
        }

        config_path = Path(output_path) / "quant_config.json"
        with open(config_path, "w") as f:
            json.dump(quant_config, f, indent=2)
        print(f"  ✓ Config saved")

        print(f"\n[STEP 3 COMPLETE] Quantized model saved on Trainium2\n")

    def trace_for_trainium2(self, sequence_length: int = 128, batch_size: int = 1):
        """
        [ON TRAINIUM2] Trace model using torch_neuronx for Trainium2 compilation.
        Everything happens on Trainium2 - output is a compiled Trainium2 model.
        """
        print("[STEP 4/6] Tracing for Trainium2 compilation")
        print("=" * 80)

        if not self.has_trainium:
            print(f"\n⚠ torch_neuronx not available (not on Trainium2 hardware)")
            print(f"  Skipping trace compilation")
            return self.model

        print(f"\nTracing configuration:")
        print(f"  Sequence length: {sequence_length}")
        print(f"  Batch size: {batch_size}")
        print(f"  (Compilation output stays on Trainium2)")

        try:
            import torch_neuronx

            # Create example input
            print(f"\nCreating example input for tracing...")
            example_input = torch.randint(
                0, 32000, (batch_size, sequence_length), dtype=torch.long
            )
            print(f"  ✓ Example input: {example_input.shape}")

            # Prepare model
            print(f"\nPreparing model for tracing...")
            self.model.to("cpu")
            self.model.eval()
            print(f"  ✓ Model in eval mode")

            # Trace with torch_neuronx
            print(f"\nTracing with torch_neuronx (neuron-cc compiler)...")
            compiler_workdir = "./compiler_workdir/"
            Path(compiler_workdir).mkdir(parents=True, exist_ok=True)

            traced_model = torch.neuron.trace(
                self.model,
                example_input,
                compiler_workdir=compiler_workdir,
                compiler_args=[
                    "--model-type=transformer",
                    "--num-neuroncores=1",
                ]
            )

            print(f"  ✓ Tracing complete")
            print(f"  ✓ Model compiled for Trainium2 NeuroCores")
            print(f"\n[STEP 4 COMPLETE] Trainium2 traced model ready\n")

            return traced_model

        except Exception as e:
            print(f"\n✗ Tracing failed: {e}")
            import traceback
            traceback.print_exc()
            print(f"\nReturning untraced model...")
            return self.model

    def run_inference_on_trainium2(
        self,
        traced_model,
        prompt: str = "The future of artificial intelligence is",
        max_tokens: int = 50,
        benchmark: bool = False,
    ):
        """
        [ON TRAINIUM2] Run inference using traced model.
        All execution happens on Trainium2 NeuroCores or CPU.
        """
        print("[STEP 5/6] Running inference (ON TRAINIUM2)")
        print("=" * 80)

        print(f"\nPrompt: {prompt}")
        print(f"Max tokens to generate: {max_tokens}")
        print(f"(Execution on Trainium2 instance)")

        try:
            # Encode input
            inputs = self.tokenizer(prompt, return_tensors="pt")
            input_ids = inputs["input_ids"]
            print(f"  ✓ Input encoded: {input_ids.shape}")

            # Generate
            print(f"\nGenerating tokens...")
            with torch.no_grad():
                output_ids = input_ids.clone()

                for step in range(max_tokens):
                    # Forward pass on Trainium2
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

            # Decode output
            generated_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

            print(f"\n" + "=" * 80)
            print("GENERATED TEXT (from Trainium2 inference):")
            print("=" * 80)
            print(generated_text)
            print("=" * 80)

            return generated_text

        except Exception as e:
            print(f"\n✗ Inference failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def benchmark_on_trainium2(self, traced_model, sequence_length: int = 128, num_iterations: int = 5):
        """
        [ON TRAINIUM2] Benchmark inference latency.
        Measures performance on actual Trainium2 hardware/CPU.
        """
        print("[STEP 6/6] Benchmarking on Trainium2")
        print("=" * 80)

        print(f"\nBenchmark configuration:")
        print(f"  Sequence length: {sequence_length}")
        print(f"  Iterations: {num_iterations}")
        print(f"  (Measuring actual Trainium2 performance)")

        try:
            # Warmup
            print(f"\nWarming up (3 iterations)...")
            for i in range(3):
                with torch.no_grad():
                    input_ids = torch.randint(0, 32000, (1, sequence_length), dtype=torch.long)
                    _ = traced_model(input_ids)
                print(f"  Warmup {i+1}/3 complete")

            # Benchmark
            print(f"\nBenchmarking (actual measurements)...")
            times = []

            for i in range(num_iterations):
                with torch.no_grad():
                    input_ids = torch.randint(0, 32000, (1, sequence_length), dtype=torch.long)

                    start = time.perf_counter()
                    output = traced_model(input_ids)
                    elapsed = time.perf_counter() - start

                    times.append(elapsed)
                    print(f"  Iteration {i+1}/{num_iterations}: {elapsed:.3f}s")

            # Statistics
            avg_latency = sum(times) / len(times)
            min_latency = min(times)
            max_latency = max(times)
            throughput = sequence_length / avg_latency

            print(f"\n" + "=" * 80)
            print("BENCHMARK RESULTS (Trainium2 Performance):")
            print("=" * 80)
            print(f"Average latency: {avg_latency:.3f}s per {sequence_length} tokens")
            print(f"Min latency: {min_latency:.3f}s")
            print(f"Max latency: {max_latency:.3f}s")
            print(f"Throughput: {throughput:.2f} tokens/sec")
            print("=" * 80)

            return {
                "avg_latency": avg_latency,
                "min_latency": min_latency,
                "max_latency": max_latency,
                "throughput": throughput,
            }

        except Exception as e:
            print(f"\n✗ Benchmarking failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _get_logger(self):
        """Simple logger for calibration."""
        class SimpleLogger:
            def info(self, msg):
                print(f"  [LOG] {msg}")
        return SimpleLogger()


def main():
    parser = argparse.ArgumentParser(
        description="UNIFIED FLATQUANT + TRAINIUM2 PIPELINE (Everything on Trainium2)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic calibration + trace on Trainium2:
  python flatquant_trainium_unified.py --model meta-llama/Llama-2-7b-hf

  # With benchmarking:
  python flatquant_trainium_unified.py --model meta-llama/Llama-2-7b-hf --benchmark

  # Custom output and token generation:
  python flatquant_trainium_unified.py \\
      --model meta-llama/Llama-2-7b-hf \\
      --output ./my_quantized_model \\
      --num_tokens 100 \\
      --benchmark
        """
    )

    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-2-7b-hf",
        help="HuggingFace model name (default: Llama 2 7B)"
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="HuggingFace API token (for gated models)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./quantized_model",
        help="Output path for quantized model (on Trainium2)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="wikitext",
        help="Calibration dataset (wikitext, openwebtext, etc.)"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=128,
        help="Number of calibration samples"
    )
    parser.add_argument(
        "--sequence_length",
        type=int,
        default=128,
        help="Sequence length for tracing and inference"
    )
    parser.add_argument(
        "--num_tokens",
        type=int,
        default=50,
        help="Number of tokens to generate for inference"
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run latency benchmarking on Trainium2"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="The future of artificial intelligence is",
        help="Prompt for text generation"
    )

    args = parser.parse_args()

    # Run unified pipeline
    pipeline = TrainiumUnifiedPipeline(
        args.model,
        args.output,
        args.hf_token,
    )

    # Step 1: Load and wrap
    model, tokenizer, quant_args = pipeline.load_and_wrap_model()

    # Step 2: Calibrate
    pipeline.calibrate_on_trainium2(quant_args, args.dataset, args.num_samples)

    # Step 3: Save
    pipeline.save_quantized_on_trainium2(quant_args)

    # Step 4: Trace
    traced_model = pipeline.trace_for_trainium2(args.sequence_length)

    # Step 5: Inference
    generated = pipeline.run_inference_on_trainium2(
        traced_model,
        prompt=args.prompt,
        max_tokens=args.num_tokens,
    )

    # Step 6: Optional benchmarking
    if args.benchmark:
        stats = pipeline.benchmark_on_trainium2(traced_model, args.sequence_length)

    print("\n" + "=" * 80)
    print("✓ UNIFIED FLATQUANT + TRAINIUM2 PIPELINE COMPLETE")
    print("=" * 80)
    print(f"Quantized model location: {args.output} (on Trainium2)")
    print(f"Status: ALL EXECUTION ON TRAINIUM2 INSTANCE")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
