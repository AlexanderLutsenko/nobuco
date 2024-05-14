#!/bin/bash

# Run `chmod +x unittest.sh` first

# Define the environment name
ENV_NAME="unittest_env"

# Remove the previous virtual environment to ensure a fresh setup
echo "Removing any existing virtual environment..."
rm -rf $ENV_NAME

# Create a new virtual environment
echo "Creating a new virtual environment..."
python3 -m venv $ENV_NAME

# Activate the virtual environment
echo "Activating the virtual environment..."
source $ENV_NAME/bin/activate

# Upgrade pip to its latest version
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies from requirements.txt
echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt

# Run unit tests
echo "Running unit tests..."
python -m unittest

# Deactivate the virtual environment
echo "Deactivating the virtual environment..."
deactivate
