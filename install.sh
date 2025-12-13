#!/bin/bash

# Define the environment name
ENV_NAME="simeco_env"

# Create the conda environment if it doesn't exist
if ! conda info --envs | grep -q "^${ENV_NAME} "; then
    echo "Creating conda environment ${ENV_NAME}..."
    conda create -n ${ENV_NAME} python=3.10 -y
fi

# Activate the conda environment
echo "Activating environment ${ENV_NAME}..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ${ENV_NAME}

# Upgrade pip (optional)
pip install --upgrade pip

# Install pip dependencies from requirements.txt
echo "Installing pip dependencies..."
pip install -r requirements.txt

# Now install pointnet2_ops
echo "Installing pointnet2_ops..."
cd extensions/pointnet2_ops_lib && python setup.py install && cd ../../

# Now install chamfer_dist
echo "Installing chamfer_dist..."
cd extensions/chamfer_dist && python setup.py install && cd ../../