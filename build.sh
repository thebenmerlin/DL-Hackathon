#!/bin/bash
# Build script for Render deployment
# This script runs during deployment to set up the environment

echo "===== Building Image Caption Generator ====="

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Download models if they don't exist
echo "Checking for pre-trained models..."
if [ ! -f "models/encoder.pth" ] || [ ! -f "models/decoder.pth" ]; then
    echo "Models not found. Initializing..."
    mkdir -p models
    python train.py
else
    echo "✓ Models already exist"
fi

echo "===== Build Complete ====="
