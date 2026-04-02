#!/usr/bin/env python3
"""
Convert HuggingFace quantized model weights to Neuron format.

This script:
1. Loads the quantized model from a checkpoint
2. Extracts the state dict
3. Maps HF keys to Neuron key format (if needed)
4. Saves in Neuron-compatible format
"""

import sys
import torch
from pathlib import Path
from typing import Dict, Set, Tuple

# Add parent paths
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, '/home/ubuntu/flatquant-trainium/FlatQuantBundled')


class QuantizedWeightConverter:
    """Converts quantized model weights to Neuron format"""

    def __init__(self, quantized_model_path: str, output_path: str):
        """
        Args:
            quantized_model_path: Path to quantized model checkpoint
            output_path: Path to save Neuron weights
        """
        self.quantized_model_path = Path(quantized_model_path)
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)

        self.hf_state_dict = None
        self.neuron_state_dict = None

    def load_quantized_state_dict(self) -> Dict[str, torch.Tensor]:
        """Load state dict from quantized model"""
        print(f"Loading quantized model state dict from {self.quantized_model_path}")

        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            str(self.quantized_model_path),
            torch_dtype=torch.float16,
            device_map="cpu",
        )

        self.hf_state_dict = model.state_dict()
        print(f"✓ Loaded {len(self.hf_state_dict)} keys from quantized model")

        return self.hf_state_dict

    def analyze_key_mapping(self) -> Tuple[Set[str], Set[str]]:
        """Analyze key structure in state dict"""
        print("\nAnalyzing state dict keys...")

        # Group keys by prefix
        prefixes = {}
        for key in self.hf_state_dict.keys():
            prefix = key.split('.')[0]
            if prefix not in prefixes:
                prefixes[prefix] = []
            prefixes[prefix].append(key)

        print(f"Found {len(prefixes)} key prefixes:")
        for prefix in sorted(prefixes.keys()):
            print(f"  {prefix}: {len(prefixes[prefix])} keys")

        # Sample some keys
        print("\nSample keys from state dict:")
        for key in sorted(self.hf_state_dict.keys())[:10]:
            shape = self.hf_state_dict[key].shape
            dtype = self.hf_state_dict[key].dtype
            print(f"  {key}: {shape} ({dtype})")

        return set(prefixes.keys()), set(self.hf_state_dict.keys())

    def map_hf_to_neuron(self) -> Dict[str, torch.Tensor]:
        """
        Map HuggingFace keys to Neuron keys.

        For standard Llama models, HF and Neuron key formats are similar,
        so mostly a 1-to-1 mapping. Custom mappings can be added if needed.
        """
        print("\nMapping HuggingFace keys to Neuron format...")

        self.neuron_state_dict = {}

        # Define key mappings (HF -> Neuron)
        key_mappings = {
            # Model embeddings
            "model.embed_tokens": "model.embed_tokens",
            # Transformer layers
            "model.layers": "model.layers",
            # Output norm and head
            "model.norm": "model.norm",
            "lm_head": "lm_head",
        }

        for hf_key, tensor in self.hf_state_dict.items():
            # For now, use 1-to-1 mapping (HF and Neuron use same key format for Llama)
            neuron_key = hf_key

            self.neuron_state_dict[neuron_key] = tensor

        print(f"✓ Mapped {len(self.neuron_state_dict)} keys")

        return self.neuron_state_dict

    def validate_state_dict(self) -> bool:
        """Validate that all required keys are present"""
        print("\nValidating state dict...")

        # Check for required top-level components
        required_prefixes = [
            "model.embed_tokens",
            "model.layers",
            "model.norm",
            "lm_head",
        ]

        missing = []
        for prefix in required_prefixes:
            found = any(k.startswith(prefix) for k in self.neuron_state_dict.keys())
            if found:
                print(f"  ✓ {prefix}")
            else:
                print(f"  ✗ {prefix}")
                missing.append(prefix)

        # Check layer count
        layer_indices = set()
        for key in self.neuron_state_dict.keys():
            if "model.layers." in key:
                parts = key.split(".")
                try:
                    idx = int(parts[2])
                    layer_indices.add(idx)
                except (ValueError, IndexError):
                    pass

        num_layers = len(layer_indices)
        print(f"\n  Found {num_layers} decoder layers")

        if missing:
            print(f"\n✗ Validation failed: missing {len(missing)} components")
            return False

        print("\n✓ Validation passed")
        return True

    def save_neuron_weights(self, format: str = "safetensors") -> Path:
        """
        Save Neuron-format weights.

        Args:
            format: Save format ("safetensors" or "torch")

        Returns:
            Path to saved weights
        """
        print(f"\nSaving Neuron weights as {format}...")

        if format == "safetensors":
            try:
                from safetensors.torch import save_file

                output_file = self.output_path / "model.safetensors"
                save_file(self.neuron_state_dict, str(output_file))
                print(f"✓ Weights saved to {output_file}")
                return output_file
            except ImportError:
                print("Warning: safetensors not available, using torch format")
                format = "torch"

        if format == "torch":
            output_file = self.output_path / "pytorch_model.bin"
            torch.save(self.neuron_state_dict, str(output_file))
            print(f"✓ Weights saved to {output_file}")
            return output_file

    def save_metadata(self):
        """Save conversion metadata"""
        import json

        metadata = {
            "source_model": str(self.quantized_model_path),
            "num_keys": len(self.neuron_state_dict),
            "key_sample": list(self.neuron_state_dict.keys())[:5],
            "conversion_type": "hf_to_neuron",
        }

        metadata_file = self.output_path / "conversion_metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"✓ Metadata saved to {metadata_file}")

    def convert(self) -> bool:
        """Run full conversion pipeline"""
        print("=" * 60)
        print("HuggingFace to Neuron Weight Conversion")
        print("=" * 60)

        try:
            # Load
            self.load_quantized_state_dict()

            # Analyze
            self.analyze_key_mapping()

            # Map
            self.map_hf_to_neuron()

            # Validate
            if not self.validate_state_dict():
                print("\n✗ Conversion failed validation")
                return False

            # Save
            self.save_neuron_weights(format="safetensors")
            self.save_metadata()

            print("\n" + "=" * 60)
            print("✓ Conversion complete!")
            print(f"  Output directory: {self.output_path}")
            print("=" * 60)

            return True

        except Exception as e:
            print(f"\n✗ Conversion failed: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert quantized model weights to Neuron format"
    )
    parser.add_argument(
        "--quantized_model",
        type=str,
        required=True,
        help="Path to quantized model checkpoint",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for Neuron weights",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="safetensors",
        choices=["safetensors", "torch"],
        help="Save format",
    )

    args = parser.parse_args()

    converter = QuantizedWeightConverter(args.quantized_model, args.output)
    success = converter.convert()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
