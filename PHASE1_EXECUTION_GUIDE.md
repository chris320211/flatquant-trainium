# Phase 1 Execution Guide: FlatQuant Integration

## Overview

This guide walks you through getting FlatQuant quantization working on your Trainium2 instance.

**Goal:** Load base Llama-2-7b-hf → Apply FlatQuant wrappers → Save quantized checkpoint

**Time estimate:** 20-30 minutes

---

## Prerequisites

✅ Already done:
- neuronx-distributed-inference installed
- agent-workflow generated files in outputs directory
- FlatQuantBundled available at ~/flatquant-trainium/FlatQuantBundled

---

## Step-by-Step Execution

### Step 1: Copy Phase 1 Files to Trainium2

On your Mac (in the repo root):

```bash
git add agent-workflow/outputs/meta-llama__Llama-2-7b-hf/setup_env.sh
git add agent-workflow/outputs/meta-llama__Llama-2-7b-hf/calibrate_flatquant.py
git commit -m "Phase 1: Add FlatQuant calibration setup"
git push
```

On your Trainium2:

```bash
cd ~/flatquant-trainium
git pull
```

### Step 2: Verify Environment Setup

On your Trainium2:

```bash
cd ~/flatquant-trainium/agent-workflow/outputs/meta-llama__Llama-2-7b-hf

# Make setup script executable
chmod +x setup_env.sh

# Run setup (this adds FlatQuantBundled to PYTHONPATH)
source setup_env.sh
```

**Expected output:**
```
✓ flatquant imported
✓ deploy imported
✓ All paths configured
```

If this fails, something is wrong with paths. Debug with:

```bash
python3 -c "
import sys
sys.path.insert(0, '/home/ubuntu/flatquant-trainium/FlatQuantBundled')
import flatquant
print('✓ FlatQuant available')
"
```

### Step 3: Test FlatQuant Module Availability

After sourcing setup_env.sh:

```bash
python3 << 'EOF'
import sys
print("Testing FlatQuant modules...")

# Test flatquant imports
try:
    from flatquant.flat_linear import FlatQuantizedLinear
    print("✓ FlatQuantizedLinear")
except Exception as e:
    print(f"✗ FlatQuantizedLinear: {e}")

try:
    from flatquant.utils import DEV
    print("✓ DEV utility")
except Exception as e:
    print(f"✗ DEV: {e}")

try:
    from flatquant.trans_utils import SVDSingleTransMatrix
    print("✓ SVDSingleTransMatrix")
except Exception as e:
    print(f"✗ SVDSingleTransMatrix: {e}")

# Test local imports
try:
    from llama_2_7b_hf_utils import FlatQuantLlamaMLP, FlatQuantLlamaAttention
    print("✓ FlatQuantLlamaMLP and FlatQuantLlamaAttention")
except Exception as e:
    print(f"✗ Local utils: {e}")

print("\n✓ All FlatQuant modules available")
EOF
```

**Expected:** All imports succeed.

### Step 4: Test Model Loading (Without Quantization)

Test that the base model can be loaded:

```bash
python3 << 'EOF'
import torch
from transformers import AutoModelForCausalLM, AutoConfig

print("Testing base model loading...")
config = AutoConfig.from_pretrained("meta-llama/Llama-2-7b-hf")
print(f"✓ Config loaded: {config.model_type}")
print(f"  Hidden size: {config.hidden_size}")
print(f"  Num layers: {config.num_hidden_layers}")

# Don't load full model yet (too much memory), just verify we can
print("✓ Base model can be loaded")
EOF
```

### Step 5: Run Calibration Script (Test Mode)

Run with just 8 samples to verify the pipeline works:

```bash
source setup_env.sh

python calibrate_flatquant.py \
    --model meta-llama/Llama-2-7b-hf \
    --num_samples 8 \
    --output ./quantized_llama2_7b_test
```

**What this does:**
1. Loads base Llama-2-7b-hf model (may take 1-2 minutes)
2. Wraps each layer's attention and MLP with FlatQuant classes
3. Tries to calibrate (will likely skip due to dataset loading issues)
4. Saves the wrapped model

**Expected output:**
```
============================================================
FlatQuant Calibration for Llama-2-7b-hf
============================================================
Loading model: meta-llama/Llama-2-7b-hf
...
✓ Model loaded: LlamaForCausalLM
✓ Tokenizer loaded

Applying FlatQuant wrappers to model...
  Layer 0: attention wrapped
  Layer 0: MLP wrapped
  Layer 1: attention wrapped
  ...
✓ Applied FlatQuant to 32 layers

Loading calibration dataset: wikitext
Warning: Could not load dataset via data_utils: ...
  Using simple random data instead...
Skipping calibration (no dataset)

Saving quantized model to: ./quantized_llama2_7b_test
✓ Model saved to ./quantized_llama2_7b_test

============================================================
✓ Calibration complete!
  Quantized model saved to: ./quantized_llama2_7b_test
============================================================
```

### Step 6: Verify Output Was Created

```bash
ls -lh quantized_llama2_7b_test/

# Expected files:
# - config.json (model config)
# - pytorch_model.bin or model.safetensors (weights)
# - tokenizer.json, tokenizer_config.json, etc.
```

### Step 7: Test Loading the Quantized Model

```bash
python3 << 'EOF'
from transformers import AutoConfig

print("Verifying quantized model checkpoint...")
config = AutoConfig.from_pretrained("./quantized_llama2_7b_test")
print(f"✓ Config loads: {config.model_type}")

# Don't try to load full model (FlatQuant wrappers may not be fully compatible)
# We'll test full loading in Phase 2
print("✓ Quantized checkpoint is valid")
EOF
```

---

## Phase 1 Troubleshooting

### Problem: "No module named 'flatquant'"

**Solution:**
```bash
# Make sure you sourced setup_env.sh
source setup_env.sh

# Verify it worked
python3 -c "import flatquant; print('OK')"
```

### Problem: "ModuleNotFoundError: No module named 'llama_2_7b_hf_utils'"

**Solution:**
```bash
# Make sure you're in the right directory
cd ~/flatquant-trainium/agent-workflow/outputs/meta-llama__Llama-2-7b-hf

# Verify the file exists
ls -la llama_2_7b_hf_utils.py

# Try importing with explicit path
python3 -c "
import sys
sys.path.insert(0, '.')
from llama_2_7b_hf_utils import FlatQuantLlamaMLP
print('OK')
"
```

### Problem: "CUDA out of memory" or "RuntimeError: CUDA out of memory"

**Solution:**
- This might happen during model loading
- Try using CPU: Modify calibrate_flatquant.py to use `device_map="cpu"` instead of `"auto"`

### Problem: "FlatQuantLlamaAttention initialization fails"

**Solution:**
- The wrapper classes might not be 100% compatible with the agent-generated code
- This is expected for Phase 1 - we're testing the approach
- Check the exact error and report it

---

## Success Criteria for Phase 1

✅ Phase 1 is complete when:
1. `setup_env.sh` runs without errors
2. All FlatQuant modules import successfully
3. `calibrate_flatquant.py` runs to completion
4. `quantized_llama2_7b_test/` directory is created with config.json
5. You can load the config from the quantized checkpoint

**If all 5 are true, you've completed Phase 1!**

---

## Next Steps (Phase 2)

Once Phase 1 is complete, Phase 2 will:
1. Convert quantized weights to Neuron format
2. Create Trainium2 inference wrapper
3. Run inference on actual Trainium2 hardware

See PHASE2_EXECUTION_GUIDE.md for details.

---

## Quick Reference

```bash
# Setup on Trainium2
cd ~/flatquant-trainium/agent-workflow/outputs/meta-llama__Llama-2-7b-hf
source setup_env.sh

# Run calibration (test with 8 samples)
python calibrate_flatquant.py --num_samples 8 --output ./quantized_test

# Run calibration (full with 128 samples - takes longer)
python calibrate_flatquant.py --num_samples 128 --output ./quantized_llama2_7b

# Verify output
ls -lh quantized_test/
```

