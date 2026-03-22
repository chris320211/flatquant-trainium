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
python verify_environment.py
```

This will check:
- All dependencies are installed
- CUDA is available
- GPU is accessible
- FlatQuant modules loaded correctly

Expected output: "SUCCESS: Environment is ready for FlatQuant!"

### 5. Load Pre-Quantized Model

Now load the FlatQuant pre-quantized LLaMA-2-7B model from HuggingFace:

```bash
python load_prequantized_model.py
```

This script:
- Downloads the W4A4KV4 quantized LLaMA-2-7B model (if not cached)
- Loads the model with FlatQuant kernels
- Runs a test inference to validate everything works
- Shows model info and generation output

**Note:** First run compiles FlatQuant CUDA kernels (takes ~30 seconds), subsequent runs are fast.

## What's Next

Once the pre-quantized model runs successfully:

1. ✅ GPU instance configured
2. ✅ CUDA working
3. ✅ FlatQuant installed
4. ✅ Pre-quantized model loaded and tested
5. **Next:** Port the model to AWS Trainium

**The FlatQuant model is now ready for Trainium translation!**

## Expected Output

```
================================================================================
FlatQuant Environment Validation
================================================================================

1. Checking dependencies...
   ✓ torch: 2.x.x
   ✓ transformers: 4.x.x
   ✓ scipy: 1.x.x
   ✓ flatquant: installed

2. Checking CUDA...
   CUDA available: True
   CUDA version: 12.0
   GPU device: Tesla T4
   GPU memory: 15.89 GB

3. Checking FlatQuant modules...
   ✓ FlatQuantizedLinear
   ✓ get_model
   ✓ load_flat_matrices

4. Testing GPU tensor operations...
   ✓ GPU tensor operation successful
   Result shape: torch.Size([100, 100]), device: cuda:0

================================================================================
SUCCESS: Environment is ready for FlatQuant!
================================================================================
```
