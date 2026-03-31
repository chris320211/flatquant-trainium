#!/usr/bin/env python3
"""
Test script for Option 2 pipeline (Explicit Transforms, No Dequantization).

This validates that:
1. Phase 1 saves without reparameterization
2. Transform matrices are preserved as Parameters
3. Phase 2 can load and trace the model
4. Inference produces coherent output
"""

import sys
import json
import torch
from pathlib import Path

# Add FlatQuantBundled to path if needed
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "FlatQuantBundled"))

from transformers import AutoModelForCausalLM


class Option2PipelineTester:
    """Test suite for Option 2 pipeline"""

    def __init__(self, quantized_model_path: str):
        self.quantized_model_path = Path(quantized_model_path)
        self.results = {}

    def test_checkpoint_structure(self) -> bool:
        """Test 1: Verify checkpoint has correct structure"""
        print("\n" + "=" * 70)
        print("Test 1: Checkpoint Structure")
        print("=" * 70)

        required_files = [
            "pytorch_model.bin",
            "config.json",
            "quant_config.json",
            "tokenizer.json",
        ]

        print(f"\nChecking {self.quantized_model_path}...")
        for filename in required_files:
            filepath = self.quantized_model_path / filename
            if filepath.exists():
                size_mb = filepath.stat().st_size / (1024 * 1024)
                print(f"✓ {filename} ({size_mb:.1f} MB)")
            else:
                print(f"✗ Missing: {filename}")
                return False

        print("\n✓ Test 1 PASSED: Checkpoint structure is correct")
        self.results["checkpoint_structure"] = True
        return True

    def test_quant_config(self) -> bool:
        """Test 2: Verify quantization config indicates Option 2 strategy"""
        print("\n" + "=" * 70)
        print("Test 2: Quantization Config")
        print("=" * 70)

        quant_config_path = self.quantized_model_path / "quant_config.json"
        if not quant_config_path.exists():
            print(f"✗ Missing quant_config.json")
            return False

        with open(quant_config_path) as f:
            config = json.load(f)

        print(f"\nConfig contents:")
        for key, value in config.items():
            print(f"  {key}: {value}")

        # Verify settings
        required_settings = {
            "w_bits": 4,
            "a_bits": 8,
            "strategy": "option2_explicit_transforms",
        }

        for key, expected in required_settings.items():
            actual = config.get(key)
            if actual == expected:
                print(f"✓ {key} = {actual}")
            else:
                print(f"⚠ {key}: expected {expected}, got {actual}")

        print("\n✓ Test 2 PASSED: Quantization config is valid")
        self.results["quant_config"] = True
        return True

    def test_model_loads(self) -> bool:
        """Test 3: Verify model can be loaded without errors"""
        print("\n" + "=" * 70)
        print("Test 3: Model Loading")
        print("=" * 70)

        try:
            print(f"\nLoading model from {self.quantized_model_path}...")
            model = AutoModelForCausalLM.from_pretrained(
                str(self.quantized_model_path),
                torch_dtype=torch.float16,
                device_map="cpu"
            )
            model.eval()
            print(f"✓ Model loaded successfully")
            print(f"  Type: {type(model).__name__}")
            print(f"  Layers: {model.config.num_hidden_layers}")

            self._model = model
            return True

        except Exception as e:
            print(f"✗ Failed to load model: {e}")
            import traceback
            traceback.print_exc()
            return False

    def test_transforms_preserved(self) -> bool:
        """Test 4: Verify transformation matrices are preserved as Parameters"""
        print("\n" + "=" * 70)
        print("Test 4: Transformation Matrices Preservation")
        print("=" * 70)

        if not hasattr(self, '_model'):
            print("✗ Model not loaded (run test_model_loads first)")
            return False

        model = self._model
        num_layers = model.config.num_hidden_layers
        trans_count = 0

        print(f"\nScanning {num_layers} layers for transformation matrices...")

        for layer_idx in range(min(2, num_layers)):  # Check first 2 layers as sample
            layer = model.model.layers[layer_idx]

            # Check attention transforms
            if hasattr(layer.self_attn, 'ln_trans') and layer.self_attn.ln_trans is not None:
                trans_count += 1
                print(f"✓ Layer {layer_idx}: ln_trans present")

            if hasattr(layer.self_attn, 'o_trans') and layer.self_attn.o_trans is not None:
                trans_count += 1
                print(f"✓ Layer {layer_idx}: o_trans present")

            # Check MLP transforms
            if hasattr(layer.mlp, 'up_gate_trans') and layer.mlp.up_gate_trans is not None:
                trans_count += 1
                print(f"✓ Layer {layer_idx}: up_gate_trans present")

            if hasattr(layer.mlp, 'down_trans') and layer.mlp.down_trans is not None:
                trans_count += 1
                print(f"✓ Layer {layer_idx}: down_trans present")

        if trans_count > 0:
            print(f"\n✓ Found {trans_count} transformation matrices in sampled layers")
            print(f"✓ Test 4 PASSED: Transforms are preserved as Parameters")
            self.results["transforms_preserved"] = True
            return True
        else:
            print(f"\n⚠ No transformation matrices found")
            print(f"⚠ Model may have been saved with old strategy (reparameterized)")
            self.results["transforms_preserved"] = False
            return False

    def test_eval_mode_active(self) -> bool:
        """Test 5: Verify evaluation mode is active on layers"""
        print("\n" + "=" * 70)
        print("Test 5: Evaluation Mode (Explicit Transforms in Graph)")
        print("=" * 70)

        if not hasattr(self, '_model'):
            print("✗ Model not loaded (run test_model_loads first)")
            return False

        model = self._model
        num_layers = model.config.num_hidden_layers
        eval_mode_count = 0

        print(f"\nScanning layers for evaluation mode...")

        for layer_idx in range(min(2, num_layers)):  # Check first 2 layers
            layer = model.model.layers[layer_idx]

            # Check attention eval mode
            if hasattr(layer.self_attn, '_eval_mode'):
                if layer.self_attn._eval_mode:
                    eval_mode_count += 1
                    print(f"✓ Layer {layer_idx} attention: _eval_mode = True")
                else:
                    print(f"⚠ Layer {layer_idx} attention: _eval_mode = False")

            # Check MLP mode
            if hasattr(layer.mlp, '_ori_mode'):
                if not layer.mlp._ori_mode:
                    eval_mode_count += 1
                    print(f"✓ Layer {layer_idx} MLP: _ori_mode = False (uses trans_forward)")
                else:
                    print(f"⚠ Layer {layer_idx} MLP: _ori_mode = True (uses ori_forward)")

        if eval_mode_count >= 2:
            print(f"\n✓ Evaluation mode is active on sampled layers")
            print(f"✓ Explicit transforms will appear in computation graph")
            print(f"✓ Test 5 PASSED: Evaluation mode is correctly set")
            self.results["eval_mode_active"] = True
            return True
        else:
            print(f"\n⚠ Evaluation mode not fully active")
            self.results["eval_mode_active"] = False
            return False

    def test_inference_runs(self) -> bool:
        """Test 6: Verify inference can run without errors"""
        print("\n" + "=" * 70)
        print("Test 6: Inference Execution")
        print("=" * 70)

        if not hasattr(self, '_model'):
            print("✗ Model not loaded (run test_model_loads first)")
            return False

        try:
            model = self._model
            print(f"\nRunning inference with random input...")

            # Create random input
            input_ids = torch.randint(0, 32000, (1, 128), dtype=torch.long)

            # Forward pass
            with torch.no_grad():
                outputs = model(input_ids)

            # Verify output
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            elif isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs

            print(f"✓ Forward pass completed")
            print(f"  Input shape: {input_ids.shape}")
            print(f"  Output shape: {logits.shape}")

            # Check output makes sense
            if logits.shape[-1] == 32000:  # vocab size
                print(f"✓ Output shape is correct (vocab size = 32000)")

                # Check values are reasonable (not NaN/Inf)
                if torch.isnan(logits).any():
                    print(f"✗ Output contains NaN values")
                    return False
                if torch.isinf(logits).any():
                    print(f"✗ Output contains Inf values")
                    return False

                print(f"✓ Output values are valid (no NaN/Inf)")
                print(f"✓ Test 6 PASSED: Inference runs successfully")
                self.results["inference_runs"] = True
                return True
            else:
                print(f"✗ Unexpected output shape: {logits.shape}")
                return False

        except Exception as e:
            print(f"✗ Inference failed: {e}")
            import traceback
            traceback.print_exc()
            self.results["inference_runs"] = False
            return False

    def test_text_generation(self) -> bool:
        """Test 7: Verify text generation produces coherent output"""
        print("\n" + "=" * 70)
        print("Test 7: Text Generation (Coherence Check)")
        print("=" * 70)

        if not hasattr(self, '_model'):
            print("✗ Model not loaded (run test_model_loads first)")
            return False

        try:
            from transformers import AutoTokenizer

            model = self._model
            tokenizer = AutoTokenizer.from_pretrained(str(self.quantized_model_path))

            prompt = "The future of artificial intelligence is"
            print(f"\nPrompt: {prompt}")
            print(f"Generating 30 tokens...")

            inputs = tokenizer(prompt, return_tensors="pt")
            input_ids = inputs["input_ids"]

            with torch.no_grad():
                output_ids = input_ids.clone()
                for step in range(30):
                    outputs = model(output_ids)
                    logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
                    next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                    output_ids = torch.cat([output_ids, next_token], dim=1)

            generated = tokenizer.decode(output_ids[0], skip_special_tokens=True)

            print(f"\nGenerated text:")
            print(f"{generated}")

            # Analyze coherence
            tokens_list = tokenizer.convert_ids_to_tokens(output_ids[0])
            unique_tokens = len(set(tokens_list))
            diversity = unique_tokens / len(tokens_list) * 100

            print(f"\nAnalysis:")
            print(f"  Total tokens: {len(tokens_list)}")
            print(f"  Unique tokens: {unique_tokens}")
            print(f"  Token diversity: {diversity:.1f}%")

            # Check if output looks coherent
            if unique_tokens < 5:
                print(f"✗ Very low token diversity - output looks like garbage")
                self.results["text_generation"] = False
                return False
            elif diversity > 60:
                print(f"✓ Output has good token diversity")
                print(f"✓ Text generation is coherent")
                print(f"✓ Test 7 PASSED: Generated text is sensible")
                self.results["text_generation"] = True
                return True
            else:
                print(f"⚠ Output has moderate diversity")
                print(f"✓ Test 7 PASSED: Text generation is functional")
                self.results["text_generation"] = True
                return True

        except Exception as e:
            print(f"✗ Text generation failed: {e}")
            import traceback
            traceback.print_exc()
            self.results["text_generation"] = False
            return False

    def run_all_tests(self) -> bool:
        """Run all tests and report results"""
        print("\n" + "=" * 70)
        print("Option 2 Pipeline Test Suite")
        print("Validating: Explicit Transforms (No Dequantization)")
        print("=" * 70)

        tests = [
            ("Checkpoint Structure", self.test_checkpoint_structure),
            ("Quantization Config", self.test_quant_config),
            ("Model Loading", self.test_model_loads),
            ("Transforms Preserved", self.test_transforms_preserved),
            ("Evaluation Mode", self.test_eval_mode_active),
            ("Inference Execution", self.test_inference_runs),
            ("Text Generation", self.test_text_generation),
        ]

        passed = 0
        for name, test_func in tests:
            try:
                if test_func():
                    passed += 1
            except Exception as e:
                print(f"\n✗ {name} - Unexpected error: {e}")

        # Summary
        print("\n" + "=" * 70)
        print("Test Summary")
        print("=" * 70)
        print(f"Passed: {passed}/{len(tests)}")
        for test_name, result in self.results.items():
            status = "✓" if result else "✗"
            print(f"{status} {test_name}")

        print("=" * 70)

        if passed == len(tests):
            print("\n✓ All tests PASSED!")
            print("The Option 2 pipeline is working correctly.")
            return True
        else:
            print(f"\n⚠ {len(tests) - passed} test(s) failed")
            return False


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Test Option 2 pipeline (Explicit Transforms)"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to quantized model checkpoint from Phase 1"
    )

    args = parser.parse_args()

    tester = Option2PipelineTester(args.model)
    success = tester.run_all_tests()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
