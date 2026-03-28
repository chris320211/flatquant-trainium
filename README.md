# FlatQuant for Trainium2

FlatQuant quantization adapted to run on AWS Trainium2 (NeuronCore-v2), with an agent workflow that automatically ports any HuggingFace causal LM to FlatQuant and generates Trainium-ready inference code.

## What this repo does

1. **FlatQuant (adapted)** — The FlatQuant training and deployment stack with CUDA/Triton kernels replaced by pure PyTorch FP8 kernels compatible with Trainium2's XLA/Neuron compiler.
2. **Porting agent** — A LangGraph agent that reads a HuggingFace model ID, extracts its architecture, and generates all FlatQuant wrapper code and Trainium inference boilerplate for that model.
3. **Pre-quantized reference** — Scripts for loading and verifying a pre-quantized W4A4KV4 checkpoint on a GPU as a reference point before Trainium deployment.

## Repository structure

```
flatquant-trainium/
  README.md                       this file
  .gitignore
  .claude/                        agent skills (Trainium model translation)
  agent-workflow/                 LangGraph porting agent
    requirements.txt
    agent/
      main.py                     entry point
      graph.py                    linear pipeline: arch → ref_reader → codegen → registration → validation
      state.py, prompts.py, tools.py, llm.py
      nodes/                      one module per pipeline step
  pre-quantized/                  reference for pre-quantized checkpoint loading
    llama2-7b/
  FlatQuantBundled/               FlatQuant library (adapted for Trainium; vendored copy)
    setup.py                      optional CUDA build; Hadamard transform install
    requirements.txt
    main.py                       FlatQuant training entry point
    flatquant/                    core algorithm (quantization, transforms, model tools)
      model_tools/                per-model utilities (llama, qwen, deepseek, ...)
    deploy/                       inference stack
      __init__.py                 lazy CUDA import (works without GPU)
      functional/                 online transforms, quantization helpers
      nn/                         Linear4bit, fp8_utils, Quantizer, OnlineTrans
      kernels/
        pytorch/                  FP8 PyTorch kernels (Trainium path)
        *.py / *.cu               original Triton/CUDA kernels (reference)
      transformers/               Llama modeling, KV cache
    tests/                        Mac/CPU smoke tests
    third-party/
      fast-hadamard-transform/    Hadamard transform (lazy-imported)
    scripts/                      training shell scripts per model
    benchmarks/                   kernel and model benchmarks
    vllm_custom/                  vLLM serving integration
```

## Quantization format

| Layer | Format | Note |
|---|---|---|
| Activations | FP8 (`torch.float8_e4m3fn`) | Per-row scale, output of `kron_matmul` / `block_matmul` |
| Weights (stored) | INT4 packed `uint8` | Expanded to BF16/FP16 for `F.linear` on Trainium |
| KV cache | INT4 asymmetric | `kv_cache.py` |

## Running the porting agent

The agent reads a HuggingFace model, generates FlatQuant wrapper code, and writes output files to `agent-workflow/outputs/<model-slug>/`.

```bash
cd agent-workflow
pip install -r requirements.txt

# Copy and fill in your Anthropic key
cp .env.example .env   # or create .env with ANTHROPIC_API_KEY=sk-...

python agent/main.py
# Model to port FlatQuant to: meta-llama/Llama-2-7b-hf
```

Generated files appear under `agent-workflow/outputs/<slug>/`:
- `<slug>_utils.py` — FlatQuant wrappers (training)
- `calibrate_<slug>.py` — calibration script
- `quant_config_<slug>.py` — quantization config
- `modeling_<slug>.py` — Trainium inference model
- `patch_<slug>.py` — module registration patch

## Running tests (Mac / CPU)

Tests run without CUDA or a Trainium instance.

```bash
cd FlatQuantBundled
pip install torch  # >= 2.1 for FP8
python -m unittest discover -s tests -v -p "test_mac*.py"
```

Six tests cover FP8 dequant, scale alignment, `sym_quant` (PyTorch fallback), `Linear4bit` with INT4 activations, and `Linear4bit` with FP8 activations.

## Trainium2 deployment

Goal: run the adapted FlatQuant stack (FP8 PyTorch kernels + `Linear4bit`) on Neuron / XLA, then compile and serve the full model with NeuronX Distributed Inference (NxDI) when you are ready.

### 1. Environment (on the instance)

On a Trainium2 instance (e.g. `trn2.48xlarge`), use the preinstalled Neuron PyTorch venv (path may vary slightly by AMI):

```bash
source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
cd flatquant-trainium/FlatQuantBundled
pip install -e third-party/fast-hadamard-transform
pip install -e .    # no CUDA extension on Trainium; PyTorch path only
```

### 2. Smoke test on Neuron / XLA

This exercises `Linear4bit` FP8 and `kron_matmul_pytorch` on the XLA device (default). Omit `--cpu` on the instance:

```bash
python scripts/trainium_smoke.py
```

For a quick local check without Neuron:

```bash
python scripts/trainium_smoke.py --cpu
```

### 3. Full model inference (NxDI)

The Neuron compile / inference path follows **NeuronX Distributed Inference**. The agent workflow’s generated `modeling_<slug>.py` under `agent-workflow/outputs/<slug>/` is the starting point: wire it into NxDI’s `NeuronConfig` / parallel layers, then compile and serve (see the Neuron SDK documentation for `neuronx_distributed_inference`). Sketch:

```python
# Example — see generated modeling_<slug>.py for your model
import torch_neuronx
from modeling_myllama import FlatQuantMyLlamaForCausalLM

model = FlatQuantMyLlamaForCausalLM.from_pretrained(...)
# Compile with torch.jit.trace or neuronx_distributed_inference APIs
```

### FP8 on Trainium2

Trainium2 (NeuronCore-v2) supports `cFP8` natively. PyTorch `torch.float8_e4m3fn` activations produced by the FP8 kernels (`kron_matmul_pytorch`, `block_matmul_pytorch`) are the correct format for XLA lowering to hardware FP8.

## Key files changed for Trainium compatibility

| File | Change |
|---|---|
| `FlatQuantBundled/deploy/__init__.py` | Lazy `deploy._CUDA` import; PyTorch `sym_quant` fallback |
| `FlatQuantBundled/deploy/nn/linear.py` | `Linear4bit` FP8 and INT4 PyTorch paths (no CUDA required) |
| `FlatQuantBundled/deploy/nn/fp8_utils.py` | FP8 dequant, INT4 weight unpack, scale alignment |
| `FlatQuantBundled/deploy/functional/online_trans.py` | Lazy `fast_hadamard_transform` import; uses PyTorch kernels |
| `FlatQuantBundled/deploy/kernels/pytorch/kron_matmul_pytorch.py` | FP8 `a @ b @ c` kernel |
| `FlatQuantBundled/deploy/kernels/pytorch/block_matmul_pytorch.py` | FP8 `b @ c` kernel |
| `FlatQuantBundled/deploy/functional/quantization.py` | Fixed `torch.tensor` warning in `get_minq_maxq` |
