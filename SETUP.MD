SSH into the TRN2 instance

git clone https://github.com/chris320211/flatquant-trainium
cd flatquant-trainium

# Update system packages (optional but recommended)
sudo apt-get update
sudo apt-get install -y python3 python3-pip 

# Create virtual environment
python3 -m venv trainium_env
source trainium_env/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel
pip install -r agent-workflow/requirements.txt