# Image Caption Generator - CNN + LSTM

Automatic caption generation for images using Deep Learning with a modern web interface.

## 🚀 Quick Start (3 Minutes)

### 1. Install Dependencies
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Initialize Models (Downloads ResNet-50 ~100MB)
```bash
python train.py
```

### 3. Start Server
```bash
python main.py
```

**Frontend**: http://localhost:8000  
**API Docs**: http://localhost:8000/docs

### 4. Test with Demo
```bash
# In another terminal
source venv/bin/activate
python demo.py
```

## 🌐 Deploy to Render (5 Minutes)

See full guide: [DEPLOYMENT.md](DEPLOYMENT.md)

**Quick Deploy:**
1. Push to GitHub
2. Connect to Render
3. Set build command: `chmod +x build.sh && ./build.sh`
4. Set start command: `python main.py`
5. Deploy!

Your app will be live at: `https://your-app.onrender.com`

## 📡 API Usage

### Generate Caption for Image
```bash
curl -X POST http://localhost:8000/caption \
  -F "file=@your_image.jpg"
```

**Response:**
```json
{
  "caption": "a person in an outdoor setting with natural surroundings",
  "inference_time_ms": 55.47,
  "model": "CNN (ResNet-50) + LSTM",
  "generation_method": "cnn-outdoor"
}
```

### Generate Captions for Multiple Images
```bash
curl -X POST http://localhost:8000/caption/batch \
  -F "files=@image1.jpg" -F "files=@image2.jpg"
```

### Check Model Info
```bash
curl http://localhost:8000/info
```

## 🏗️ Architecture

### CNN (Convolutional Neural Network)
- **Model**: ResNet-50 (pre-trained on ImageNet)
- **Purpose**: Extract visual features from images
- **Output**: 256-dimensional feature vector
- **How it works**: 
  - Removes final classification layer
  - Uses global average pooling features
  - Projects to embedding space

### RNN (Recurrent Neural Network) - LSTM
- **Model**: Long Short-Term Memory network
- **Purpose**: Generate sequential text descriptions
- **Architecture**:
  - Embedding layer: 256 dimensions
  - LSTM hidden state: 512 units
  - Output vocabulary: 126 words
  - Max sequence length: 50 tokens

### How It Works
1. **Image Upload** → User uploads image via API
2. **Preprocessing** → Resize to 224×224, normalize
3. **Feature Extraction (CNN)** → ResNet-50 extracts 256-D features
4. **Caption Generation**:
   - **LSTM Decoder**: Generates caption word-by-word
   - **CNN Analysis**: Analyzes feature patterns for scene type
   - **Smart Fallback**: Uses CNN features if LSTM needs training
5. **Response** → Returns caption with inference metrics

## 📁 Project Structure

```
DL Hackathon/
├── main.py              # FastAPI backend server + frontend serving
├── cnn_model.py         # CNN feature extractor (ResNet-50)
├── rnn_model.py         # LSTM caption generator
├── vocabulary.py        # Vocabulary management
├── train.py             # Model initialization script
├── demo.py              # Quick demo script
├── test_api.py          # API test suite
├── requirements.txt     # Python dependencies
├── build.sh             # Render build script
├── runtime.txt          # Python version for Render
├── render.yaml          # Render configuration
├── .renderignore        # Files to exclude from deployment
├── README.md            # This file
├── DEPLOYMENT.md        # Detailed deployment guide
│
├── templates/           # HTML templates
│   └── index.html       # Main frontend page
│
├── static/              # Static assets
│   ├── css/
│   │   └── style.css    # Frontend styles
│   └── js/
│       └── app.js       # Frontend JavaScript
│
└── models/              # Saved models (auto-created)
    ├── encoder.pth      # CNN weights
    ├── decoder.pth      # LSTM weights
    └── vocab.pkl        # Vocabulary
```

## 🎯 Current Capabilities

✅ **Working**: Modern responsive frontend with dark/light theme  
✅ **Working**: Drag-and-drop image upload  
✅ **Working**: CNN feature extraction (ResNet-50)  
✅ **Working**: LSTM decoder architecture  
✅ **Working**: Gemini-style sidebar with caption & description  
✅ **Working**: API endpoints (single & batch)  
✅ **Working**: Scene type detection (outdoor/indoor/nature/person)  
✅ **Working**: Fast inference (~30-80ms per image)  
✅ **Working**: Recent captions saved in localStorage  
✅ **Ready**: Production deployment configuration for Render  

## 🚧 Next Steps for Improvement

1. **Train on Real Data**: Use Flickr8k/Flickr30k/MS-COCO datasets
2. **Better Captions**: Train LSTM with real image-caption pairs
3. **Attention Mechanism**: Add visual attention for better accuracy
4. **Beam Search**: Improve generation quality
5. **Confidence Scores**: Add caption confidence metrics
6. **Fine-tuning**: Adapt to specific domains

## 📊 Training on Real Data

To train with real data:
1. Download Flickr8k or MS-COCO dataset
2. Place images in `data/images/`
3. Create caption file in `data/captions.txt`
4. Run: `python train.py`

## 🛠️ Tech Stack

- **Backend**: FastAPI (Python)
- **Deep Learning**: PyTorch
- **CNN Architecture**: ResNet-50 (pre-trained)
- **RNN Architecture**: LSTM
- **Image Processing**: PIL/Pillow
- **Server**: Uvicorn (ASGI)
