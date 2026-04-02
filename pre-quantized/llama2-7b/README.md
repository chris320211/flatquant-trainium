# LLaMA-2-7B FlatQuant Checkpoint Loader

Simple script to load and validate FlatQuant W4A4KV4 quantized LLaMA-2-7B model.

## Prerequisites

- AWS EC2 GPU instance (g4dn.xlarge or g5.xlarge)
- Deep Learning AMI with CUDA pre-installed
- Python 3.8+

## Quick Start on AWS EC2

### 1. Launch EC2 Instance

**IMPORTANT:** Use GPU AMI, NOT Neuron AMI!

1. Search for AMI: `Deep Learning AMI GPU PyTorch`
2. Select: **"Deep Learning OSS Nvidia Driver AMI GPU PyTorch"** (Ubuntu 22.04)
   - ❌ DO NOT select "Deep Learning AMI Neuron" (for Trainium/Inferentia, not NVIDIA GPUs)
   - ✅ Must say "GPU" or "Nvidia Driver" in the name
3. Instance type: `g5.2xlarge` (~$1.20/hour, recommended) or `g4dn.xlarge` (~$0.50/hour, slower)
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
git clone https://github.com/chris320211/flatquant-trainium.git
cd flatquant-trainium

# Create virtual environment (for Ubuntu 24.04 compatibility)
python3 -m venv venv
source venv/bin/activate

# Install core dependencies
pip install --upgrade pip wheel cmake
pip install torch transformers==4.45.0 scipy

# Clone and install FlatQuant (takes 10-20 minutes)
git clone https://github.com/ruikangliu/FlatQuant.git
cd FlatQuant
pip install -e . --no-build-isolation

# Install additional required dependencies
pip install fast-hadamard-transform

# Set PYTHONPATH so model can find FlatQuant modules
export PYTHONPATH="${PYTHONPATH}:${HOME}/flatquant-trainium/FlatQuantBundled"
```

**Important:**
- Always activate venv: `source ~/flatquant-trainium/venv/bin/activate`
- FlatQuant compilation takes 10-20 minutes (compiling CUDA kernels) - this is normal!
- Must use `transformers==4.45.0` (newer versions have compatibility issues)
- Must set PYTHONPATH or model loading will fail
- To make PYTHONPATH permanent, add to `~/.bashrc`:
  ```bash
  echo 'export PYTHONPATH="${PYTHONPATH}:${HOME}/flatquant-trainium/FlatQuantBundled"' >> ~/.bashrc
  ```

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
