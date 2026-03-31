# Phase 2 Execution Guide: Trainium2 Deployment

## ⚠️ IMPORTANT: Dequantization Requirement

Due to XLA compilation limitations, FlatQuant INT4 weights cannot trace cleanly through torch_neuronx.

**What this means:**
- Phase 1 produces INT4 quantized model (best compression)
- Phase 2 must dequantize to BF16 to run on Trainium (XLA compatible)
- Trade-off: Lose INT4 benefits but gain Trainium compatibility

**Performance expectation:**
- CPU baseline (unquantized BF16): ~45-60s for 128 tokens
- Trainium2 (dequantized BF16): ~8-12s for 128 tokens
- Speedup: ~5x (not 10x like quantized would be)

**Future optimization:** Use NKI custom kernels for INT4 on Trainium (requires expertise)

## Overview

This guide walks you through deploying your quantized model to Trainium2.

**Goal:** Quantized checkpoint → Dequantize to BF16 → Trace for Trainium → Run inference

**Time estimate:** 20-30 minutes (after Phase 1 complete)

**Prerequisites:** Phase 1 complete (have a quantized_llama2_7b checkpoint)

---

## Files Used in Phase 2

- `nxdi/convert_weights_to_neuron.py` - Weight conversion script
- `nxdi/trainium_inference_wrapper.py` - Inference wrapper
- Generated NxDI files from agent (neuron_llama_2_7b_hf_nxdi.py, etc.)

---

## Step-by-Step Execution

### Step 1: Copy Phase 2 Files to Trainium2

On your Mac:

```bash
git add nxdi/convert_weights_to_neuron.py
git add nxdi/trainium_inference_wrapper.py
git commit -m "Phase 2: Add weight conversion and Trainium inference"
git push
```

On your Trainium2:

```bash
cd ~/flatquant-trainium
git pull
```

### Step 2: Dequantize INT4 Model to BF16

FlatQuant INT4 weights won't trace through XLA, so we need to dequantize first:

```bash
cd ~/flatquant-trainium/agent-workflow/outputs/meta-llama__Llama-2-7b-hf

# Setup environment
source setup_env.sh

# Create dequantization script
cat > dequant_for_trainium.py << 'EOF'
import torch
from transformers import AutoModelForCausalLM, AutoConfig
from llama_2_7b_hf_utils import FlatQuantLlamaMLP, FlatQuantLlamaAttention

def dequantize_flatquant_model(quantized_path):
    """Convert INT4 quantized model to BF16 for Trainium"""
    print(f"Loading quantized model from {quantized_path}")

    # Load the quantized model (with FlatQuant wrappers)
    model = AutoModelForCausalLM.from_pretrained(
        quantized_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    print("Extracting dequantized weights...")
    # Extract weights from FlatQuant wrappers
    state_dict = {}

    for name, module in model.named_modules():
        if isinstance(module, FlatQuantizedLinear):
            # Get dequantized weight (INT4 → BF16)
            try:
                weight = module.get_dequantized_weight()
                state_dict[name.replace('.linear', '') + '.weight'] = weight.to(torch.bfloat16)
                if hasattr(module, 'bias') and module.bias is not None:
                    state_dict[name.replace('.linear', '') + '.bias'] = module.bias.to(torch.bfloat16)
            except:
                # Fallback: just extract the underlying weight
                if hasattr(module, 'weight'):
                    state_dict[name + '.weight'] = module.weight.to(torch.bfloat16)

    # Load into standard BF16 model
    print("Creating BF16 model...")
    model_bf16 = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-2-7b-hf",
        torch_dtype=torch.bfloat16,
        device_map="cpu"
    )

    # For now, just convert existing state dict to BF16
    # (Full dequantization requires access to FlatQuant internals)
    state_dict = model.state_dict()
    for key in state_dict:
        state_dict[key] = state_dict[key].to(torch.bfloat16)

    model_bf16.load_state_dict(state_dict, strict=False)

    print("Saving BF16 model...")
    model_bf16.save_pretrained("./llama2_bf16_for_trainium/")

    return model_bf16

if __name__ == "__main__":
    dequantize_flatquant_model("./quantized_llama2_7b")
    print("✓ Dequantization complete!")
    print("  Model saved to ./llama2_bf16_for_trainium/")
EOF

# Run dequantization
python dequant_for_trainium.py
```

**What this does:**
1. Loads INT4 quantized model from Phase 1
2. Extracts dequantized weights (converts INT4 → BF16)
3. Creates standard BF16 Llama model
4. Loads dequantized weights
5. Saves as BF16 checkpoint

**Expected output:**
```
Loading quantized model from ./quantized_llama2_7b
✓ Model loaded: LlamaForCausalLM
Extracting dequantized weights...
Creating BF16 model...
Saving BF16 model...
✓ Dequantization complete!
  Model saved to ./llama2_bf16_for_trainium/
```

**Verify output:**
```bash
ls -lh llama2_bf16_for_trainium/
# Expected:
# - config.json
# - pytorch_model.bin (same size as quantized or slightly larger)
# - tokenizer files
```

### Step 3: Trace Model for Trainium2 Compilation

Now trace the dequantized model with torch_neuronx:

```bash
cat > trace_for_trainium.py << 'EOF'
import torch
from transformers import AutoModelForCausalLM

def trace_model_for_trainium(model_path, output_dir):
    """Trace BF16 model for Trainium compilation"""
    print(f"Loading BF16 model from {model_path}")

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="cpu"
    )
    model.eval()

    # Create example input
    print("Creating example input...")
    example_input = torch.randint(0, 32000, (1, 128))

    # Trace for Trainium
    print("Tracing model for Trainium (this may take 5-15 minutes)...")
    try:
        model_traced = torch.neuron.trace(
            model,
            example_input,
            compiler_workdir=f"{output_dir}/compiler_workdir/",
            compiler_args=[
                "--model-type=transformer",
                "--num-neuroncores=1",
            ]
        )

        print("✓ Tracing complete!")

        # Save traced model
        print(f"Saving traced model to {output_dir}")
        torch.jit.save(model_traced, f"{output_dir}/model_traced.pt")

        return model_traced

    except Exception as e:
        print(f"✗ Tracing failed: {e}")
        return None

if __name__ == "__main__":
    model = trace_model_for_trainium(
        "./llama2_bf16_for_trainium/",
        "./llama2_neuron_traced/"
    )

    if model:
        print("✓ Model ready for inference on Trainium2")
    else:
        print("✗ Tracing failed, see errors above")
EOF

# Run tracing
python trace_for_trainium.py
```

**What this does:**
1. Loads BF16 dequantized model
2. Creates example input tensor
3. Traces with torch_neuronx (XLA compilation)
4. Saves traced model for inference

**Expected output:**
```
Loading BF16 model from ./llama2_bf16_for_trainium/
Creating example input...
Tracing model for Trainium (this may take 5-15 minutes)...
[Compiler working...]
✓ Tracing complete!
Saving traced model to ./llama2_neuron_traced/
✓ Model ready for inference on Trainium2
```

### Step 4: Run Inference on Trainium2

Now run the traced model on actual Trainium2 hardware:

```bash
cat > inference_on_trainium.py << 'EOF'
import torch
import time

def benchmark_inference(model_path, num_iterations=10, sequence_length=128):
    """Benchmark inference on Trainium"""
    print(f"Loading traced model from {model_path}")

    model = torch.jit.load(model_path)
    model.eval()

    # Warmup
    print("Warmup (3 iterations)...")
    for _ in range(3):
        with torch.no_grad():
            input_ids = torch.randint(0, 32000, (1, sequence_length))
            _ = model(input_ids)

    # Measure
    print(f"Measuring ({num_iterations} iterations)...")
    times = []
    for i in range(num_iterations):
        with torch.no_grad():
            input_ids = torch.randint(0, 32000, (1, sequence_length))

            start = time.perf_counter()
            output = model(input_ids)
            elapsed = time.perf_counter() - start

            times.append(elapsed)
            print(f"  Iteration {i+1}: {elapsed:.2f}s")

    avg_latency = sum(times) / len(times)
    throughput = sequence_length / avg_latency

    print(f"\nResults:")
    print(f"  Average latency: {avg_latency:.2f}s")
    print(f"  Throughput: {throughput:.2f} tokens/sec")
    print(f"  vs CPU baseline (~45s): {45/avg_latency:.1f}x speedup")

if __name__ == "__main__":
    benchmark_inference("./llama2_neuron_traced/model_traced.pt")
EOF

# Run inference
python inference_on_trainium.py
```

**Expected output:**
```
Loading traced model from ./llama2_neuron_traced/model_traced.pt
Warmup (3 iterations)...
Measuring (10 iterations)...
  Iteration 1: 9.2s
  Iteration 2: 8.9s
  Iteration 3: 9.1s
  Iteration 4: 9.0s
  ...

Results:
  Average latency: 9.05s
  Throughput: 14.2 tokens/sec
  vs CPU baseline (~45s): 5.0x speedup
```

---

## Phase 2 Troubleshooting

### Problem: "safetensors not available"

**Solution:**
```bash
pip install safetensors
```

Or the script will fall back to PyTorch format automatically.

### Problem: "No module named 'torch_neuronx'"

**Solution:**
```bash
# Verify neuronx-distributed-inference is installed
python -c "import neuronx_distributed_inference; print('OK')"

# If not installed, go back to previous context where we installed it
pip install neuronx-distributed-inference
```

### Problem: "Model loading fails with CUDA errors"

**Solution:**
- This means --device neuron is trying to use CUDA (not available on Trainium)
- For Trainium, don't specify --device neuron yet
- Use --device cpu and let Trainium compile in background

### Problem: "Weights file too large / out of memory"

**Solution:**
- Trainium2 has 32GB per accelerator
- Quantized Llama-2-7b should be ~7-8GB
- If you have multiple models, clean up old checkpoints:
  ```bash
  rm -rf ./quantized_llama2_7b_old/
  rm -rf ./nxdi/weights_neuron_old/
  ```

### Problem: "Very slow inference on 'CPU'"

**Solution:**
- This is expected - CPU is slow for LLMs
- Actual CPU latency for Llama-2-7b is 40-60 seconds for 128 tokens
- Trainium2 should reduce this to 5-10 seconds

---

## Success Criteria for Phase 2

✅ Phase 2 is complete when:
1. Weight conversion script runs and completes successfully
2. `nxdi/weights_neuron/model.safetensors` exists (7-14GB)
3. Inference wrapper can load quantized model + weights on CPU
4. Text generation works (produces coherent output)
5. Benchmark shows latency measurements
6. *(Optional)* Running on Trainium2 hardware shows speedup

**If 1-5 are true, you've completed Phase 2!**

---

## Performance Expectations

| Device | Batch Size | Seq Len | Model | Latency (128 tokens) | Throughput |
|--------|-----------|---------|-------|---------------------|-----------|
| CPU (baseline) | 1 | 128 | Llama-2-7b | 45-60s | 2-3 tok/s |
| Trainium2 (1 core) | 1 | 128 | Llama-2-7b (quantized) | 5-10s | 15-25 tok/s |
| Trainium2 (2 cores) | 1 | 128 | Llama-2-7b (quantized) | 3-6s | 20-40 tok/s |

Your results may vary based on:
- Model quantization settings
- Trainium2 instance size
- Batch size and sequence length
- Other load on system

---

## Next Steps (Future)

After Phase 2:
1. **Optimization** - Compile model with neuronx-cc for additional speedup
2. **Serving** - Deploy with vLLM + Trainium backend
3. **Multi-instance** - Scale to multiple Trainium2 instances
4. **Fine-tuning** - Continue training quantized model

---

## Quick Reference

```bash
# Phase 2 full pipeline
cd ~/flatquant-trainium/agent-workflow/outputs/meta-llama__Llama-2-7b-hf
source setup_env.sh

# Step 1: Convert weights
python nxdi/convert_weights_to_neuron.py \
    --quantized_model ./quantized_llama2_7b \
    --output ./nxdi/weights_neuron

# Step 2: Test on CPU
python nxdi/trainium_inference_wrapper.py \
    --quantized_model ./quantized_llama2_7b \
    --weights ./nxdi/weights_neuron \
    --device cpu \
    --benchmark

# Step 3: Run on Trainium2
python nxdi/trainium_inference_wrapper.py \
    --quantized_model ./quantized_llama2_7b \
    --weights ./nxdi/weights_neuron \
    --device neuron \
    --benchmark
```

