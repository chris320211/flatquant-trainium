# FlatQuant + Trainium2 Deployment Tools

Complete workflow for quantizing a model with FlatQuant and deploying it to AWS Trainium2 hardware.

## Overview

This toolkit provides a two-phase pipeline:

- **Phase 1: FlatQuant Quantization** - Convert a model to INT4 using FlatQuant
- **Phase 2: Trainium2 Deployment** - Dequantize, trace, and run on Trainium2 hardware

## Phase 1: FlatQuant Quantization

### Prerequisites
- HuggingFace model (e.g., `meta-llama/Llama-2-7b-hf`)
- FlatQuantBundled library installed at `/home/ubuntu/flatquant-trainium/FlatQuantBundled`
- Agent-generated model-specific wrappers (e.g., `llama_2_7b_hf_utils.py` in `agent-workflow/outputs/meta-llama__Llama-2-7b-hf/`)

### Step 1: Setup Environment

```bash
cd agent-workflow/tools/phase1/
source setup_env.sh
```

This script:
- Adds FlatQuantBundled to your Python path
- Verifies imports work
- Sets required environment variables

### Step 2: Run Quantization

```bash
python phase1/calibrate_flatquant.py \
    --model meta-llama/Llama-2-7b-hf \
    --output ./quantized_model \
    --num_samples 128
```

**Arguments:**
- `--model`: HuggingFace model name or path
- `--output`: Where to save the quantized checkpoint
- `--num_samples`: Calibration samples (8 for testing, 128 for production)
- `--dataset`: Calibration dataset (default: `wikitext`)
- `--hf_token`: HuggingFace API token (if needed)

**Output:**
- `./quantized_model/` directory with INT4 quantized weights

**Timing:**
- ~20-30 minutes on Trainium2 (depending on samples)
- ~5 minutes on fast GPU

## Phase 2: Trainium2 Deployment

### Prerequisites
- Quantized model from Phase 1
- torch_neuronx (AWS Trainium compiler) - only works on Trainium2 hardware
- BF16 support

### ⚠️ Critical Note: Dequantization Requirement

**FlatQuant INT4 weights cannot trace through XLA.** Before deploying to Trainium2, you must:
1. Dequantize INT4 → BF16
2. Trace with torch_neuronx
3. Run on Trainium2

This trades INT4 compression for Trainium2 compatibility. Expected speedup: ~5x (not 5-10x).

### Step 1: Dequantize INT4 to BF16

```bash
python phase2/dequant_for_trainium.py \
    --quantized_model ./quantized_model \
    --output ./llama2_bf16_for_trainium
```

**What happens:**
- Loads INT4 quantized model
- Extracts weights to BF16 precision
- Creates standard HuggingFace BF16 Llama model
- Saves to `./llama2_bf16_for_trainium/`

**Timing:**
- ~5-10 minutes (weight extraction + conversion)

### Step 2: Trace for Trainium2

```bash
python phase2/trace_for_trainium.py \
    --model ./llama2_bf16_for_trainium \
    --output ./llama2_neuron_traced \
    --num_neuroncores 1
```

**What happens:**
- Loads BF16 model
- Creates example inputs
- Traces through torch_neuronx (XLA compilation)
- Saves TorchScript traced model

**Arguments:**
- `--model`: Path to BF16 model from Step 1
- `--output`: Where to save traced model
- `--num_neuroncores`: 1, 2, or 8 (Trainium2 has 32 cores)
- `--sequence_length`: Example input sequence length (default: 128)

**Timing:**
- **5-15 minutes** on Trainium2 (XLA compilation is slow)
- May not work on non-Trainium2 systems (torch_neuronx unavailable)

### Step 3: Run Inference & Benchmark

```bash
# Run benchmark
python phase2/inference_on_trainium.py \
    --model ./llama2_neuron_traced/model_traced.pt \
    --benchmark \
    --num_iterations 10 \
    --sequence_length 128

# Run text generation
python phase2/inference_on_trainium.py \
    --model ./llama2_neuron_traced/model_traced.pt \
    --prompt "The future of AI is" \
    --max_tokens 50
```

**Benchmark output example:**
```
Results:
Average latency: 8.05s
Throughput: 15.9 tokens/sec
Trainium speedup: 5.6x faster than CPU
```

**Text generation output example:**
```
Prompt: The future of AI is
Generated: The future of AI is incredibly exciting. With advances in machine learning...
```

## Full Workflow Example

```bash
# Setup (one time)
cd /path/to/flatquant-trainium/agent-workflow/tools/phase1/
source setup_env.sh

# Phase 1: Quantize (20-30 min)
cd /path/to/flatquant-trainium/agent-workflow/tools/
python phase1/calibrate_flatquant.py \
    --model meta-llama/Llama-2-7b-hf \
    --output ./quantized_model \
    --num_samples 128

# Phase 2: Deploy to Trainium2 (20-30 min total)
python phase2/dequant_for_trainium.py \
    --quantized_model ./quantized_model \
    --output ./llama2_bf16_for_trainium

python phase2/trace_for_trainium.py \
    --model ./llama2_bf16_for_trainium \
    --output ./llama2_neuron_traced

python phase2/inference_on_trainium.py \
    --model ./llama2_neuron_traced/model_traced.pt \
    --benchmark
```

## Troubleshooting

### Phase 1 Issues

**ImportError: No module named 'flatquant'**
```bash
cd agent-workflow/tools/phase1/
source setup_env.sh  # Always do this first!
```

**Model loading fails**
```bash
# Check HuggingFace access
python -c "from transformers import AutoConfig; \
    AutoConfig.from_pretrained('meta-llama/Llama-2-7b-hf'); \
    print('OK')"

# May need token
python phase1/calibrate_flatquant.py \
    --model meta-llama/Llama-2-7b-hf \
    --hf_token your_token_here \
    --output ./quantized_model
```

**Out of memory**
```bash
# Reduce batch size in calibration
python phase1/calibrate_flatquant.py \
    --num_samples 8 \
    --output ./quantized_model
```

### Phase 2 Issues

**ImportError: No module named 'torch_neuronx'**
- ⚠️ This only works on Trainium2 hardware
- Check: `python -c "import torch_neuronx; print('OK')"`

**Tracing takes forever (15+ min)**
- Normal! XLA compilation can take 5-15 minutes
- Don't interrupt - it will finish
- First run is slowest due to compilation

**Model loading in inference fails**
```bash
# Verify traced model exists
ls -lh ./llama2_neuron_traced/model_traced.pt

# Check file size (should be ~14GB)
```

## Architecture

```
agent-workflow/tools/
├── phase1/
│   ├── setup_env.sh              # Environment setup
│   └── calibrate_flatquant.py    # Main quantization script
│
└── phase2/
    ├── dequant_for_trainium.py   # INT4 → BF16 conversion
    ├── trace_for_trainium.py     # XLA tracing
    └── inference_on_trainium.py  # Inference & benchmark
```

**Generated artifacts (in agent-workflow/outputs/):**
```
agent-workflow/outputs/meta-llama__Llama-2-7b-hf/
├── llama_2_7b_hf_utils.py       # Agent-generated FlatQuant wrappers
├── quant_config_llama_2_7b_hf.py
├── calibrate_llama_2_7b_hf.py
└── ... (other generated files)
```

## Performance Expectations

| Stage | Hardware | Latency | Speedup |
|-------|----------|---------|---------|
| CPU Baseline | CPU | ~45s | 1.0x |
| BF16 Model | Trainium2 | ~8-10s | 4.5-5.6x |
| INT4 Model | (Not supported) | - | - |

**Note:** INT4 quantization is lost during dequantization, so we don't measure INT4 performance. If INT4 kernel optimization is critical, see `COMPREHENSIVE_FIX_PLAN.md` for NKI custom kernel approach.

## Key Decisions

### Why Dequantize?
- FlatQuant INT4 ops use custom PyTorch operations that don't trace through XLA
- XLA compilation (torch_neuronx) only understands standard PyTorch ops
- Dequantization to BF16 ensures clean tracing

### Why Not Skip Dequantization?
- Custom INT4 kernels would require NKI programming expertise
- Dequantization is simple, reliable, and documented
- BF16 still provides speedup (5x) on Trainium2

### Why PyTorch Kernels?
- Portable across different hardware
- Works on CPU for testing
- Trainium2 has native PyTorch 2.9 support
- XLA can compile standard PyTorch ops

## Next Steps

1. **Test Phase 1** on your Trainium2 with `--num_samples 8` (5 min) to verify setup
2. **Run full Phase 1** with `--num_samples 128` for production quantization
3. **Test Phase 2** with small sequence lengths (`--sequence_length 64`) first
4. **Benchmark** to measure actual Trainium2 speedup on your hardware

## References

- [FlatQuant Paper](https://arxiv.org/abs/2407.10554)
- [AWS Trainium Documentation](https://awsdocs-neuron.readthedocs-hosted.com/)
- [torch_neuronx API](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/release-notes/torch-neuronx/torch-neuronx.html)
- [COMPREHENSIVE_FIX_PLAN.md](../COMPREHENSIVE_FIX_PLAN.md) - Technical deep-dive

## Support

For issues, check:
1. `COMPREHENSIVE_FIX_PLAN.md` for technical details
2. `PHASE1_EXECUTION_GUIDE.md` for Phase 1 walkthrough
3. `PHASE2_EXECUTION_GUIDE.md` for Phase 2 walkthrough
