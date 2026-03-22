# LLaMA-2-7B FlatQuant Checkpoint Loader

Simple script to load and validate FlatQuant W4A4KV4 quantized LLaMA-2-7B model.

## Prerequisites

- AWS EC2 GPU instance (g4dn.xlarge or g5.xlarge)
- Deep Learning AMI with CUDA pre-installed
- Python 3.8+

## Quick Start on AWS EC2

### 1. Launch EC2 Instance

1. Search for AMI: `Deep Learning AMI GPU PyTorch`
2. Select: "Deep Learning AMI GPU PyTorch 2.0.1 Ubuntu 20.04" (or latest)
3. Instance type: `g4dn.xlarge` (~$0.50/hour)
4. Storage: 100 GB
5. Create/select SSH key pair
6. Launch instance

### 2. Connect to Instance

```bash
ssh -i ~/.ssh/your-key.pem ubuntu@ec2-XX-XX-XX-XX.compute-1.amazonaws.com
```

### 3. Setup Environment

```bash
# Clone the repo
git clone https://github.com/YOUR-USERNAME/flatquant-trainium.git
cd flatquant-trainium/pre-quantized/llama2-7b

# Upgrade CMake (required for FlatQuant)
pip install --upgrade cmake

# Install dependencies in order (torch first, then FlatQuant)
pip install torch transformers scipy
pip install git+https://github.com/ruikangliu/FlatQuant.git
# Note: FlatQuant compilation takes 10-20 minutes - this is normal!
```

**Important:**
- Install torch BEFORE FlatQuant
- Upgrade CMake first to avoid build errors
- FlatQuant compilation takes 10-20 minutes (compiling CUDA kernels)
- Install scipy to avoid import errors

### 4. Verify Installation

First, test that everything is installed correctly:

```bash
python load_checkpoint_simple.py
```

This will check:
- All dependencies are installed
- CUDA is available
- GPU is accessible
- FlatQuant modules loaded correctly

Expected output: "SUCCESS: Environment is ready for FlatQuant!"

### 5. Download Models (TODO - Not yet available)

You need two things:
1. **LLaMA-2-7B base model** from HuggingFace
2. **FlatQuant W4A4KV4 matrices**

```bash
# Create directories
mkdir -p modelzoo/meta-llama
mkdir -p modelzoo/flatquant/llama-2-7b/w4a4kv4

# TODO: Add instructions once model files are available
# For now, you need to:
# 1. Download LLaMA-2-7B from HuggingFace (requires license acceptance)
# 2. Obtain FlatQuant matrices from FlatQuant repo or generate them
```

## Usage

**Once you have the model files:**

```bash
python load_checkpoint.py \
  --model_path ./modelzoo/meta-llama/Llama-2-7b-hf \
  --matrix_path ./modelzoo/flatquant/llama-2-7b/w4a4kv4
```

**Note:** The script requires both `--model_path` and `--matrix_path` arguments.

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
