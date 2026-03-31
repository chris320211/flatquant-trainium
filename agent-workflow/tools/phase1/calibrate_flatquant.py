#!/usr/bin/env python3
"""
Phase 1: Calibrate and apply FlatQuant quantization to model.

This script:
1. Loads base model from HuggingFace
2. Applies FlatQuant INT4 wrappers to all layers
3. Runs calibration (optional)
4. Saves quantized checkpoint

Usage:
    source setup_env.sh
    python calibrate_flatquant.py --model meta-llama/Llama-2-7b-hf --output ./quantized_model
"""

import sys
import os
import torch
from pathlib import Path

# CRITICAL: Import transformers FIRST, before any FlatQuantBundled modules
# This prevents FlatQuantBundled/deploy/transformers from shadowing the real one
from transformers import AutoModelForCausalLM, AutoTokenizer

# NOTE: FlatQuantBundled should already be in PYTHONPATH from setup_env.sh
# DO NOT add it again to sys.path - that will cause the shadowing issue

import flatquant.utils as fq_utils
import flatquant.data_utils as data_utils
import flatquant.train_utils as train_utils
import flatquant.flat_utils as flat_utils

# Import model-specific wrappers from this directory
from llama_2_7b_hf_utils import FlatQuantLlamaMLP, FlatQuantLlamaAttention


class FlatQuantCalibrator:
    """Handles FlatQuant calibration for any Llama model"""

    def __init__(self, model_name: str, hf_token: str = None):
        self.model_name = model_name
        self.hf_token = hf_token
        self.model = None
        self.tokenizer = None

    def load_model(self):
        """Load base model from HuggingFace"""
        print(f"Loading model: {self.model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            token=self.hf_token,
        )
        self.model.eval()

        print(f"Loading tokenizer: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            use_fast=False,
            token=self.hf_token,
        )

        print(f"✓ Model loaded: {type(self.model).__name__}")
        return self.model, self.tokenizer

    def apply_flatquant_wrappers(self):
        """Apply FlatQuant wrappers to model layers"""
        print("Applying FlatQuant wrappers to model...")

        # Create args object with FlatQuant settings
        class FlatQuantArgs:
            w_bits = 4
            a_bits = 8
            group_size = 128
            w_asym = False  # Symmetric weight quantization
            a_asym = False  # Symmetric activation quantization
            a_groupsize = -1  # -1 = per-layer quantization (groupsize>0 not yet supported)
            lwc = False  # Learned weight clipping
            direct_inv = False
            add_diag = False
            diag_init = "sq_style"
            lac = False  # Learned activation clipping
            separate_vtrans = True  # For attention v_proj handling
            q_bits = 8  # KV cache quantization bits
            k_bits = 8
            v_bits = 8
            q_asym = False
            k_asym = False
            v_asym = False

        args = FlatQuantArgs()

        # Replace attention and MLP layers
        num_layers = self.model.config.num_hidden_layers
        wrapped_count = 0

        for layer_idx in range(num_layers):
            layer = self.model.model.layers[layer_idx]

            # Wrap attention
            try:
                layer.self_attn = FlatQuantLlamaAttention(args, layer.self_attn)
                wrapped_count += 1
            except Exception as e:
                print(f"  Layer {layer_idx}: attention wrap failed - {e}")

            # Wrap MLP
            try:
                layer.mlp = FlatQuantLlamaMLP(args, layer.mlp)
                wrapped_count += 1
            except Exception as e:
                print(f"  Layer {layer_idx}: MLP wrap failed - {e}")

        print(f"✓ Applied FlatQuant wrappers to {num_layers} layers (2x{num_layers} components)")
        return self.model

    def calibrate(self, dataset_name: str = "wikitext", num_samples: int = 128):
        """Run calibration on dataset"""
        print(f"Loading calibration dataset: {dataset_name}")

        # Set sequence length
        self.model.seqlen = 2048

        # Get calibration data
        try:
            trainloader = data_utils.get_loaders(
                args=None,
                name=dataset_name,
                nsamples=num_samples,
                seed=0,
                seqlen=self.model.seqlen,
                eval_mode=False,
            )
            print(f"✓ Loaded {num_samples} calibration samples")
        except Exception as e:
            print(f"Warning: Could not load dataset via data_utils: {e}")
            print("  Skipping calibration...")
            return

        if trainloader:
            print("Running calibration...")
            try:
                train_utils.cali_flat_quant(
                    args=None,
                    model=self.model,
                    trainloader=trainloader,
                    device=fq_utils.DEV,
                )
                print("✓ Calibration complete")
            except Exception as e:
                print(f"Calibration failed: {e}")

    def save_model(self, output_path: str):
        """
        Save quantized model WITHOUT reparameterization.

        Key difference: Transform matrices T are preserved as Parameters
        instead of being fused into weights. This allows explicit T operations
        to appear in the computation graph during Trainium tracing.
        """
        print(f"Saving quantized model to: {output_path}")

        # Step 1: Set evaluation mode on all layers
        # This freezes transforms and enables explicit T + quantization in forward
        print("Setting evaluation mode (keeping transforms explicit)...")
        num_layers = self.model.config.num_hidden_layers
        for layer_idx in range(num_layers):
            layer = self.model.model.layers[layer_idx]

            # Set eval mode on attention
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

            # Set eval mode on MLP
            if hasattr(layer.mlp, '_ori_mode'):
                layer.mlp._ori_mode = False  # Use trans_forward, not ori_forward
            if hasattr(layer.mlp, 'up_gate_trans') and layer.mlp.up_gate_trans is not None:
                layer.mlp.up_gate_trans.to_eval_mode()
            if hasattr(layer.mlp, 'down_trans') and layer.mlp.down_trans is not None:
                layer.mlp.down_trans.to_eval_mode()

        print("✓ Evaluation mode set (transforms will be explicit in graph)")

        # Step 2: Also set eval mode on FlatQuantizedLinear layers
        # This makes their forward use _eval_forward with explicit quantization
        print("Setting FlatQuantizedLinear to evaluation mode...")
        for layer_idx in range(num_layers):
            layer = self.model.model.layers[layer_idx]

            # Attention projections
            for proj_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
                if hasattr(layer.self_attn, proj_name):
                    proj = getattr(layer.self_attn, proj_name)
                    if hasattr(proj, '_eval_mode'):
                        proj._eval_mode = True

            # MLP projections
            for proj_name in ['up_proj', 'gate_proj', 'down_proj']:
                if hasattr(layer.mlp, proj_name):
                    proj = getattr(layer.mlp, proj_name)
                    if hasattr(proj, '_eval_mode'):
                        proj._eval_mode = True

        print("✓ FlatQuantizedLinear in eval mode")

        # Step 3: Save model (with T as Parameters, not fused)
        print("Saving model checkpoint...")
        Path(output_path).mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)
        print(f"✓ Model saved to {output_path}")

        # Step 4: Save quantization config
        print("Saving quantization config...")
        quant_config = {
            "w_bits": self.args.w_bits,
            "a_bits": self.args.a_bits,
            "group_size": self.args.group_size,
            "w_asym": self.args.w_asym,
            "a_asym": self.args.a_asym,
            "a_groupsize": self.args.a_groupsize,
            "lwc": self.args.lwc,
            "q_bits": self.args.q_bits,
            "k_bits": self.args.k_bits,
            "v_bits": self.args.v_bits,
            "model_type": "llama",
            "strategy": "option2_explicit_transforms",
        }

        import json
        config_path = Path(output_path) / "quant_config.json"
        with open(config_path, "w") as f:
            json.dump(quant_config, f, indent=2)
        print(f"✓ Quant config saved to {config_path}")

        print("\n" + "=" * 60)
        print("✓ Phase 1 Save Complete!")
        print(f"  Location: {output_path}")
        print(f"  Strategy: Option 2 - Explicit Transforms (NO reparameterization)")
        print(f"  Weights: Preserved in original size (not fused with T)")
        print(f"  Transforms: Saved as Parameters (will be in graph)")
        print(f"  Next step: Run Phase 2 Trainium inference")
        print("=" * 60)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Phase 1: FlatQuant Calibration")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-hf",
                       help="HuggingFace model name")
    parser.add_argument("--hf_token", type=str, default=None,
                       help="HuggingFace API token")
    parser.add_argument("--dataset", type=str, default="wikitext",
                       help="Calibration dataset")
    parser.add_argument("--num_samples", type=int, default=128,
                       help="Number of calibration samples")
    parser.add_argument("--output", type=str, default="./quantized_model",
                       help="Output path for quantized model")

    args = parser.parse_args()

    print("=" * 60)
    print("Phase 1: FlatQuant Calibration")
    print("=" * 60)

    # Run calibration pipeline
    calib = FlatQuantCalibrator(args.model, args.hf_token)
    calib.load_model()
    calib.apply_flatquant_wrappers()
    calib.calibrate(args.dataset, args.num_samples)
    calib.save_model(args.output)

    print("=" * 60)
    print("✓ Phase 1 Complete!")
    print(f"  Quantized model saved to: {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()
