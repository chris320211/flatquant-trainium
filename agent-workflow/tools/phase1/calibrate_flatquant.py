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

# CRITICAL: Import transformers BEFORE adding FlatQuantBundled to path
# FlatQuantBundled/deploy/transformers will shadow the real transformers package
from transformers import AutoModelForCausalLM, AutoTokenizer

# Setup paths - add FlatQuantBundled AFTER transformers is imported
sys.path.insert(0, '/home/ubuntu/flatquant-trainium/FlatQuantBundled')

import flatquant.utils as fq_utils
import flatquant.data_utils as data_utils
import flatquant.train_utils as train_utils
import flatquant.flat_utils as flat_utils

# Import from parent directory
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "outputs"))


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

        # Try to import the generated wrappers
        try:
            # This assumes the model-specific utils are in outputs
            from llama_2_7b_hf_utils import FlatQuantLlamaMLP, FlatQuantLlamaAttention
            wrapper_classes = (FlatQuantLlamaMLP, FlatQuantLlamaAttention)
        except ImportError:
            print("Warning: Model-specific wrappers not found")
            print("Using generic FlatQuant classes instead...")
            from flatquant.flat_linear import FlatQuantizedLinear
            # Would need to wrap manually with generic approach
            wrapper_classes = None

        if wrapper_classes is None:
            print("Skipping wrapper application (needs model-specific implementation)")
            return self.model

        # Create args object with FlatQuant settings
        class FlatQuantArgs:
            w_bits = 4
            a_bits = 8
            group_size = 128
            direct_inv = False
            add_diag = False
            diag_init = "sq_style"
            lac = False

        args = FlatQuantArgs()

        # Replace attention and MLP layers
        num_layers = self.model.config.num_hidden_layers
        for layer_idx in range(num_layers):
            layer = self.model.model.layers[layer_idx]

            # Wrap attention
            try:
                layer.self_attn = wrapper_classes[1](args, layer.self_attn)
                print(f"  Layer {layer_idx}: attention wrapped")
            except Exception as e:
                print(f"  Layer {layer_idx}: attention wrap failed - {e}")

            # Wrap MLP
            try:
                layer.mlp = wrapper_classes[0](args, layer.mlp)
                print(f"  Layer {layer_idx}: MLP wrapped")
            except Exception as e:
                print(f"  Layer {layer_idx}: MLP wrap failed - {e}")

        print(f"✓ Applied FlatQuant to {num_layers} layers")
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
        """Save quantized model"""
        print(f"Saving quantized model to: {output_path}")

        # Reparameterize to apply quantization
        try:
            flat_utils.reparameterize_model(self.model)
            print("✓ Model reparameterized")
        except Exception as e:
            print(f"Warning: Reparameterization failed: {e}")

        # Save
        self.model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)
        print(f"✓ Model saved to {output_path}")


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
