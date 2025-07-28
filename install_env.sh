#!/bin/bash
#
## Copyright 2025 Lin Zhang (partialspoof@gmail.com)
# 
# source ./install_env.sh

# Make sure conda is avalilabel
eval "$(conda shell.bash hook)"

env_name=wedefense
if conda info --envs | grep -q "$env_name"; then
    echo "Conda env ${env_name} already exist."
    conda activate ${env_name}
else
    echo "Create conda env $env_name"
    conda create -n ${env_name} python=3.10
    conda activate ${env_name}
    
    echo "Start to install pytorch using conda"
    conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.4 -c pytorch -c nvidia
    pip install -r requirements.txt
fi

# For managing and maintaining multi-language pre-commit hooks.
pre-commit install

## You may consider to use pip to install your env
## Note that this version cannot support codec augmentation because of version issues.
#conda create -n ${env_name} python=3.10
#conda activate ${env_name}
#pip install torch==2.1.2+cu121 torchaudio==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
#pip install -r requirements.txt


# If you got warn: ModuleNotFoundError: No module named 'whisper'
if ! python -c "import whisper" &> /dev/null; then
    echo "whisper module not found, installing openai-whisper..."
        pip install -U openai-whisper --no-cache-dir
else
    echo "whisper module is already installed."
fi

# For users in BUT's server.
if [[ $(hostname -f) == *fit.vutbr.cz ]]; then
    echo "You are working on BUT server, we will install safe_gpu."
    pip install safe_gpu
fi

