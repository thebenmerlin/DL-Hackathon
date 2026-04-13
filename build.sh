#!/bin/bash
# Build script for Render deployment
# This script runs during deployment to set up the environment

echo "===== Building Image Caption Generator ====="

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Create models directory
mkdir -p models

# Download ResNet-18 and save model weights during build
echo "Downloading and initializing models..."
python -c "
import os
os.environ['TORCH_HOME'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.torch_cache')

import torch
from cnn_model import CNNFeatureExtractor
from rnn_model import LSTMDecoderWithAttention
from vocabulary import Vocabulary

# Initialize models (downloads ResNet-18 during build, not startup)
print('Downloading ResNet-18 (this may take a minute)...')
encoder = CNNFeatureExtractor(embed_size=256)
decoder = LSTMDecoderWithAttention(vocab_size=126, embed_size=256, hidden_size=256)

# Save model weights
torch.save(encoder.state_dict(), 'models/encoder.pth')
torch.save(decoder.state_dict(), 'models/decoder.pth')
print('Models saved successfully')

# Create vocabulary
vocab = Vocabulary()
for word in ['<pad>', '<start>', '<end>', '<unk>', 'a', 'an', 'the', 'person', 'scene', 'in', 'with', 'and', 'outdoor', 'indoor', 'natural', 'setting', 'focal', 'point', 'multiple', 'visible', 'elements', 'room', 'furnishings', 'urban', 'environment', 'trees', 'greenery']:
    vocab.add_word(word)
vocab.save('models/vocab.pkl')
print('Vocabulary saved')
"

echo "===== Build Complete ====="
