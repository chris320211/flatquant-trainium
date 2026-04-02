SSH into the TRN2 instance

git clone https://github.com/chris320211/flatquant-trainium
cd flatquant-trainium

# Update system packages (optional but recommended)
sudo apt-get update
sudo apt-get install -y python3 python3-pip 

# Create virtual environment
python3 -m venv trainium_env

# Upgrade pip
pip install --upgrade pip setuptools wheel
pip install -r agent-workflow/requirements.txt

export HF_TOKEN="your_huggingface_token_here"

# Navigate to tools directory
cd agent-workflow/tools

# Source setup to set up PYTHONPATH
source setup_env.sh

# Run the end-to-end unified pipeline
python flatquant_trainium_unified.py \
    --model meta-llama/Llama-2-7b-hf \
    --output ./quantized_llama2_7b \
    --num_tokens 50 \
    --benchmark

This will run all 6 steps on Trainium2:
Load Llama2-7B
Apply FlatQuant INT4+FP8 wrappers
Calibrate and learn transform matrices
Save quantized checkpoint
Trace with torch_neuronx
Generate 50 tokens to verify coherence (this is your gibberish check)