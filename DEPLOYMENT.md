# Deploy to Render - Step by Step Guide

## Quick Deployment (5 minutes)

### Option 1: Deploy from GitHub (Recommended)

1. **Push your code to GitHub**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin YOUR_GITHUB_URL
   git push -u origin main
   ```

2. **Go to Render**
   - Visit: https://render.com
   - Sign up/Login

3. **Create New Web Service**
   - Click "New +" → "Web Service"
   - Connect your GitHub repository
   - Configure:
     - **Name**: `image-caption-generator`
     - **Region**: Choose nearest to your users
     - **Branch**: `main`
     - **Root Directory**: Leave blank
     - **Runtime**: `Python 3`
     - **Build Command**: `chmod +x build.sh && ./build.sh`
     - **Start Command**: `python main.py`

4. **Environment Variables** (Add these)
   ```
   PORT=8000
   RELOAD=false
   ```

5. **Click "Create Web Service"**
   - Render will automatically build and deploy
   - First deployment takes ~5-8 minutes (downloads ResNet-50 model)

6. **Access your app**
   - URL: `https://image-caption-generator.onrender.com`

---

### Option 2: Deploy with Render CLI

1. **Install Render CLI**
   ```bash
   npm install -g render-cli
   ```

2. **Login to Render**
   ```bash
   render login
   ```

3. **Deploy**
   ```bash
   render up
   ```

---

## Option 3: Manual Deployment

### 1. Create a Render Account
- Go to https://render.com
- Sign up with GitHub, GitLab, or email

### 2. Create New Web Service
- Dashboard → New + → Web Service
- Connect your Git repository (GitHub, GitLab, or Bitbucket)

### 3. Configure Service

**Basic Settings:**
- Name: `image-caption-generator`
- Region: Oregon / Frankfurt / Singapore (choose closest)
- Branch: `main`
- Root Directory: (leave blank)

**Build Settings:**
- Runtime: `Python 3`
- Build Command: 
  ```bash
  chmod +x build.sh && ./build.sh
  ```
- Start Command:
  ```bash
  python main.py
  ```

**Environment Variables:**
```
PORT=8000
RELOAD=false
```

### 4. Choose Instance Type
- **Free Tier**: Good for demo (may sleep after inactivity)
- **Starter ($7/mo)**: Always on, faster response
- **Standard ($25/mo)**: Production-ready

### 5. Deploy
- Click "Create Web Service"
- Watch the deployment logs
- First deploy: ~5-8 minutes

---

## Local Testing Before Deployment

```bash
# 1. Activate virtual environment
source venv/bin/activate

# 2. Run the server
python main.py

# 3. Open browser
open http://localhost:8000
```

---

## Post-Deployment Checklist

- [ ] Homepage loads at `https://your-app.onrender.com`
- [ ] Image upload works
- [ ] Caption generation returns results
- [ ] Sidebar displays caption and description
- [ ] Theme toggle works
- [ ] Recent captions saved in localStorage

---

## Troubleshooting

### Build Fails
**Problem**: Dependencies not installing
**Solution**: Check `requirements.txt` has all packages

### Models Not Found
**Problem**: `Models not found` error
**Solution**: Ensure `train.py` runs during build

### Port Error
**Problem**: Port already in use
**Solution**: Render sets PORT env variable automatically. Check `main.py` uses `os.environ.get("PORT", 8000)`

### Slow First Request
**Problem**: First request takes 30+ seconds
**Solution**: This is normal on free tier (cold start). Upgrade to paid tier for always-on.

---

## Custom Domain (Optional)

1. Go to your Render Dashboard
2. Select your service
3. Settings → Custom Domain
4. Add your domain
5. Update DNS records as instructed

---

## Environment Variables Reference

| Variable | Description | Default |
|----------|-------------|---------|
| `PORT` | Server port (set by Render) | `8000` |
| `RELOAD` | Auto-reload on code changes | `false` |

---

## Monitoring

- **Logs**: Dashboard → Your Service → Logs
- **Metrics**: Dashboard → Your Service → Metrics
- **Health Check**: `GET https://your-app.onrender.com/health`

---

## Cost Estimate

- **Free Tier**: $0 (sleeps after 15min inactivity)
- **Starter**: $7/month (always on)
- **Storage**: Models ~100MB (included in free tier)

---

## Need Help?

- Render Docs: https://render.com/docs
- FastAPI Docs: https://fastapi.tiangolo.com
- PyTorch: https://pytorch.org
