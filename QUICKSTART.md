# 🚀 Quick Reference - Image Caption Generator

## Local Development

```bash
# Start server
source venv/bin/activate
python main.py

# Access
# Frontend: http://localhost:8000
# API Docs: http://localhost:8000/docs
```

## Deploy to Render (One-Click)

1. Push to GitHub
2. Connect to Render.com
3. Settings:
   - Build: `chmod +x build.sh && ./build.sh`
   - Start: `python main.py`
   - Port: `8000` (auto-set by Render)
4. Deploy!

## Project Files

### Core Application
- `main.py` - FastAPI server + serves frontend
- `cnn_model.py` - ResNet-50 CNN feature extractor
- `rnn_model.py` - LSTM text generator
- `vocabulary.py` - Word vocabulary management
- `train.py` - Model initialization

### Frontend
- `templates/index.html` - Main webpage
- `static/css/style.css` - Styles (dark/light theme)
- `static/js/app.js` - Frontend logic

### Deployment
- `requirements.txt` - Python dependencies
- `build.sh` - Render build script
- `runtime.txt` - Python version
- `render.yaml` - Render config
- `DEPLOYMENT.md` - Full deployment guide

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Frontend webpage |
| GET | `/health` | Health check |
| POST | `/caption` | Generate caption for image |
| POST | `/caption/batch` | Batch caption generation |
| GET | `/info` | Model information |
| GET | `/docs` | Interactive API docs |

## Features

✅ CNN (ResNet-50) + LSTM architecture  
✅ Drag-and-drop image upload  
✅ Gemini-style sidebar with caption & description  
✅ Dark/Light theme toggle  
✅ Scene type detection (outdoor/indoor/nature/person)  
✅ Recent captions (localStorage)  
✅ Copy & share functionality  
✅ Production-ready for Render  
✅ Mobile responsive  

## Tech Stack

- **Backend**: FastAPI (Python 3.11+)
- **Deep Learning**: PyTorch
- **CNN**: ResNet-50 (ImageNet pre-trained)
- **RNN**: LSTM (256 embed, 512 hidden)
- **Frontend**: Vanilla JS + CSS3
- **Deployment**: Render.com

## Model Architecture

```
Image (224x224)
    ↓
[CNN: ResNet-50]
    ↓
Features (256-dim)
    ↓
[LSTM: 512 units]
    ↓
Caption (text)
```

## Quick Test

```bash
# Test with curl
curl -X POST http://localhost:8000/caption \
  -F "file=@your_image.jpg"

# Expected response
{
  "caption": "a person in an outdoor...",
  "inference_time_ms": 38.14,
  "model": "CNN (ResNet-50) + LSTM",
  "generation_method": "cnn-outdoor"
}
```
