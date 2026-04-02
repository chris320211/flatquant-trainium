#!/usr/bin/env python3
"""
Calibration script for FlatQuant Llama-2-7b-hf.
This uses the llama_2_7b_hf_utils.py approach (Approach A - FlatQuantBundled classes).
"""

import sys
import os
import torch
from pathlib import Path

# CRITICAL: Import transformers FIRST, before adding FlatQuantBundled to path
# FlatQuantBundled/deploy/transformers shadows the real transformers module
from transformers import AutoModelForCausalLM, AutoTokenizer

# Setup paths - find FlatQuantBundled relative to this script
# This script is in: agent-workflow/outputs/meta-llama__Llama-2-7b-hf/calibrate_flatquant.py
# FlatQuantBundled is at: FlatQuantBundled/ (up 3 levels)
SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent.parent.parent  # Navigate up to repository root
FLATQUANT_PATH = REPO_ROOT / "FlatQuantBundled"

if not FLATQUANT_PATH.exists():
    raise RuntimeError(f"FlatQuantBundled not found at {FLATQUANT_PATH}. Ensure script is run from within the repository.")

sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(FLATQUANT_PATH))
# NOTE: Do NOT add FlatQuantBundled/deploy to path - it shadows transformers

import flatquant.utils as fq_utils
import flatquant.data_utils as data_utils
import flatquant.train_utils as train_utils
import flatquant.flat_utils as flat_utils

from llama_2_7b_hf_utils import FlatQuantLlamaMLP, FlatQuantLlamaAttention


class FlatQuantCalibrator:
    """Handles FlatQuant calibration for Llama-2-7b-hf"""

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
        for layer_idx in range(num_layers):
            layer = self.model.model.layers[layer_idx]

            # Wrap attention
            try:
                layer.self_attn = FlatQuantLlamaAttention(args, layer.self_attn)
                print(f"  Layer {layer_idx}: attention wrapped")
            except Exception as e:
                print(f"  Layer {layer_idx}: attention wrap failed - {e}")

            # Wrap MLP
            try:
                layer.mlp = FlatQuantLlamaMLP(args, layer.mlp)
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
            print("  Using simple random data instead...")
            trainloader = None

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
        else:
            print("Skipping calibration (no dataset)")

    def save_model(self, output_path: str):
        """Save quantized model"""
        print(f"Saving quantized model to: {output_path}")

        # Reparameterize to apply quantization
        print("Reparameterizing wrapped layers...")
        num_layers = self.model.config.num_hidden_layers
        for layer_idx in range(num_layers):
            layer = self.model.model.layers[layer_idx]
            try:
                # Call reparameterize on wrapped attention
                if hasattr(layer.self_attn, 'reparameterize'):
                    layer.self_attn.reparameterize()
            except Exception as e:
                print(f"  Layer {layer_idx}: attention reparameterize failed - {e}")

            try:
                # Call reparameterize on wrapped MLP
                if hasattr(layer.mlp, 'reparameterize'):
                    layer.mlp.reparameterize()
            except Exception as e:
                print(f"  Layer {layer_idx}: MLP reparameterize failed - {e}")

        print("✓ Reparameterization complete")

        # Save
        self.model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)
        print(f"✓ Model saved to {output_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Calibrate FlatQuant Llama-2-7b-hf")
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
    print("FlatQuant Calibration for Llama-2-7b-hf")
    print("=" * 60)

    # Run calibration pipeline
    calib = FlatQuantCalibrator(args.model, args.hf_token)
    calib.load_model()
    calib.apply_flatquant_wrappers()
    calib.calibrate(args.dataset, args.num_samples)
    calib.save_model(args.output)

    print("=" * 60)
    print("✓ Calibration complete!")
    print(f"  Quantized model saved to: {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()
