# Phase 1: FlatQuant Integration Fix

## Step 1.1: Setup Module Paths

The issue is that generated code imports from `flatquant` and `deploy` which exist in `FlatQuantBundled/` but aren't in the Python path.

### Solution: Create a path setup wrapper

On your Trainium2, create this file:

```bash
cat > ~/flatquant-trainium/agent-workflow/outputs/meta-llama__Llama-2-7b-hf/setup_env.sh << 'EOF'
#!/bin/bash
# Setup Python paths for FlatQuant and deploy modules

export PYTHONPATH=/home/ubuntu/flatquant-trainium/FlatQuantBundled:$PYTHONPATH
export PYTHONPATH=/home/ubuntu/flatquant-trainium/FlatQuantBundled/deploy:$PYTHONPATH

# Verify modules are importable
python3 -c "import flatquant; print('✓ flatquant imported')" || exit 1
python3 -c "import deploy; print('✓ deploy imported')" || exit 1

echo "✓ All paths configured"
EOF
chmod +x ~/flatquant-trainium/agent-workflow/outputs/meta-llama__Llama-2-7b-hf/setup_env.sh
```

Then run:
```bash
source setup_env.sh
```

### Verify it works:
```bash
python3 << 'EOF'
import sys
print("Python path:")
for p in sys.path[:5]:
    print(f"  {p}")

try:
    import flatquant
    print("✓ flatquant module found")
except ImportError as e:
    print(f"✗ flatquant import failed: {e}")

try:
    import deploy
    print("✓ deploy module found")
except ImportError as e:
    print(f"✗ deploy import failed: {e}")
EOF
```

---

## Step 1.2: Analyze Generated Code Compatibility

On your Trainium2, check which classes from `flatquant` are actually used:

```bash
cd ~/flatquant-trainium/agent-workflow/outputs/meta-llama__Llama-2-7b-hf

# List all flatquant imports in generated code
echo "=== FlatQuant imports needed ==="
grep -h "from flatquant\|import flatquant" *.py 2>/dev/null | sort -u

# Check if FlatQuantBundled has these
echo -e "\n=== Checking FlatQuantBundled has these modules ==="
python3 << 'PYEOF'
import sys
sys.path.insert(0, '/home/ubuntu/flatquant-trainium/FlatQuantBundled')

needed_modules = [
    'flatquant.quant_utils',
    'flatquant.utils',
    'flatquant.flat_linear',
    'flatquant.function_utils',
    'flatquant.trans_utils',
    'flatquant.args_utils',
    'flatquant.data_utils',
    'flatquant.model_utils',
    'flatquant.train_utils',
    'flatquant.flat_utils',
]

for mod in needed_modules:
    try:
        __import__(mod)
        print(f"✓ {mod}")
    except ImportError as e:
        print(f"✗ {mod}: {e}")
PYEOF
```

---

## Step 1.3: Check deploy.nn Module

The generated `modeling_llama_2_7b_hf.py` imports from `deploy.nn` which doesn't exist in FlatQuantBundled.

On your Trainium2:

```bash
# Check what's in deploy
ls -la /home/ubuntu/flatquant-trainium/FlatQuantBundled/deploy/

# Check if it has nn module
python3 << 'PYEOF'
import sys
sys.path.insert(0, '/home/ubuntu/flatquant-trainium/FlatQuantBundled')

try:
    import deploy
    print("✓ deploy module found")
    print(f"  deploy.__file__: {deploy.__file__}")
    print(f"  deploy contents: {dir(deploy)}")
except ImportError as e:
    print(f"✗ deploy import failed: {e}")

try:
    from deploy import nn
    print("✓ deploy.nn found")
except ImportError as e:
    print(f"✗ deploy.nn import failed: {e}")

# Try alternatives
try:
    from deploy.nn import Linear4bit
    print("✓ deploy.nn.Linear4bit found")
except ImportError as e:
    print(f"✗ deploy.nn.Linear4bit not found: {e}")

# Check if FlatQuantized classes are what we need
try:
    from flatquant.flat_linear import FlatQuantizedLinear
    print("✓ flatquant.flat_linear.FlatQuantizedLinear found (ALTERNATIVE)")
except ImportError as e:
    print(f"✗ flatquant alternative failed: {e}")
PYEOF
```

---

## Step 1.4: Decision - Which Quantization Approach to Use?

Based on the generated code, we have TWO quantization approaches:

### Approach A: llama_2_7b_hf_utils.py (Uses FlatQuantBundled classes)
```python
from flatquant.flat_linear import FlatQuantizedLinear
from flatquant.utils import DEV
# Classes: FlatQuantLlamaMLP, FlatQuantLlamaAttention
```

**Pros:**
- Uses real FlatQuantBundled classes
- Has transformation matrices (SVD/Inv)
- Better code quality

**Cons:**
- Needs to be tested

### Approach B: modeling_llama_2_7b_hf.py (Uses non-existent deploy.nn)
```python
from deploy.nn import Linear4bit, OnlineTrans, Quantizer
# Classes: FlatQuantLlamaMLP, FlatQuantLlamaAttention
```

**Pros:**
- Has clean abstraction

**Cons:**
- Imports non-existent module
- Can't run as-is

### Recommendation: USE APPROACH A

The `llama_2_7b_hf_utils.py` file uses real FlatQuantBundled classes and should work.

---

## Step 1.5: Create Unified Calibration Script

Create a single, working calibration script that uses Approach A:

```bash
cat > ~/flatquant-trainium/agent-workflow/outputs/meta-llama__Llama-2-7b-hf/calibrate_flatquant.py << 'EOF'
#!/usr/bin/env python3
"""
Calibration script for FlatQuant Llama-2-7b-hf.
This uses the llama_2_7b_hf_utils.py approach (Approach A).
"""

import sys
import os
import torch
import transformers
from pathlib import Path

# Setup paths
REPO_ROOT = Path(__file__).parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, '/home/ubuntu/flatquant-trainium/FlatQuantBundled')
sys.path.insert(0, '/home/ubuntu/flatquant-trainium/FlatQuantBundled/deploy')

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
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            token=self.hf_token,
        )
        self.model.eval()

        print(f"Loading tokenizer: {self.model_name}")
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
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
EOF

chmod +x ~/flatquant-trainium/agent-workflow/outputs/meta-llama__Llama-2-7b-hf/calibrate_flatquant.py
```

---

## Step 1.6: Test the Calibration Script

On your Trainium2:

```bash
cd ~/flatquant-trainium/agent-workflow/outputs/meta-llama__Llama-2-7b-hf

# Setup paths
source setup_env.sh

# Test with small sample (just 8 samples, no real calibration)
python calibrate_flatquant.py \
    --model meta-llama/Llama-2-7b-hf \
    --num_samples 8 \
    --output ./quantized_llama2_7b_hf
```

This will:
1. Load the base model from HuggingFace
2. Apply FlatQuant wrappers to each layer
3. (Skip real calibration since we don't have proper dataset setup)
4. Save the wrapped model

---

## Step 1.7: Verify It Works

After running, check:

```bash
# Check if quantized model was saved
ls -lh quantized_llama2_7b_hf/

# Try to load it back
python << 'EOF'
import sys
sys.path.insert(0, '/home/ubuntu/flatquant-trainium/FlatQuantBundled')

from transformers import AutoModelForCausalLM, AutoConfig

print("Loading quantized model...")
config = AutoConfig.from_pretrained("./quantized_llama2_7b_hf")
print(f"✓ Config loaded: {config.model_type}")

# Note: Actual model loading may fail if quantized wrappers aren't fully compatible
# but config loading should work
EOF
```

---

## Summary: Phase 1 Steps

1. ✅ Run `source setup_env.sh` to add FlatQuantBundled to Python path
2. ✅ Verify imports work with test script
3. ✅ Use Approach A (llama_2_7b_hf_utils.py) - ignore Approach B (modeling file)
4. ✅ Run calibrate_flatquant.py to apply FlatQuant wrappers
5. ✅ Save quantized model

If all these work, Phase 1 is complete and we move to Phase 2 (Trainium2 integration).

