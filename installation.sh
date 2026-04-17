#!/bin/bash

apt-get update;
apt-get install git screen vim curl htop wget -y;

cd ..
python3 -m venv env;
source env/bin/activate;
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124; 
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.5.0+cu124.html
pip install transformers accelerate torcheval torchmetrics matplotlib pandas ipdb gdown notebook pyyaml openai groq openai peft immutabledict google-generativeai absl-py scikit-learn nltk ogb datasets rdkit pyyaml wandb;
pip install torch-geometric;
printf "\033c";