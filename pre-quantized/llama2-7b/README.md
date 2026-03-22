# LLaMA-2-7B FlatQuant Checkpoint Loader

Simple script to load and validate FlatQuant W4A4KV4 quantized LLaMA-2-7B model.

## Prerequisites

- CUDA-enabled GPU (FlatQuant requires CUDA)
- Python 3.8+

## Setup

1. Install dependencies:
```bash
pip install torch transformers
pip install git+https://github.com/ruikangliu/FlatQuant.git
```

Or use the requirements file:
```bash
pip install -r requirements.txt
```

**Note:** FlatQuant requires CUDA to build. Installation will fail on macOS or systems without CUDA.

## Usage

```bash
python load_checkpoint.py \
  --model_path ./modelzoo/meta-llama/Llama-2-7b-hf \
  --matrix_path ./modelzoo/flatquant/llama-2-7b/w4a4kv4
```

## What it does

The script:
1. Loads the LLaMA-2-7B base model in FP16
2. Applies FlatQuant W4A4KV4 transformation matrices
3. Prints the model architecture
4. Lists quantized layers
5. Runs a dummy forward pass to validate functionality
6. Shows sample weight dtypes to confirm INT4 storage

## Expected Output

```
================================================================================
Loading LLaMA-2-7B with FlatQuant W4A4KV4
================================================================================
Model path: ./modelzoo/meta-llama/Llama-2-7b-hf
Matrix path: ./modelzoo/flatquant/llama-2-7b/w4a4kv4
================================================================================

Loading base model in FP16...
Base model loaded: LlamaForCausalLM

Applying FlatQuant transformation from ./modelzoo/flatquant/llama-2-7b/w4a4kv4...
FlatQuant transformation applied

... [model architecture] ...

================================================================================
SUCCESS: Model loaded and forward pass completed!
================================================================================
```
