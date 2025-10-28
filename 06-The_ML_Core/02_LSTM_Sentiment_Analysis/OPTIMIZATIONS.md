# ⚡ CPU Optimization Applied

## Changes Made for Old CPU:

### 1. **Reduced Model Size**
- Embedding Dimension: 400 → **128** (68% smaller)
- Hidden Dimension: 256 → **64** (75% smaller)  
- LSTM Layers: 2 → **1** (50% less)

### 2. **Reduced Training Data**
- Training samples: 25,000 → **1,000** (96% less)
- Test samples: 25,000 → **500** (98% less)
- Sequence Length: 256 → **128** (50% shorter)

### 3. **Faster Processing**
- Epochs: 4 → **1** (75% less)
- Batch Size: 50 → **128** (2.5x larger = fewer batches)

### 4. **Estimated Training Time**
- **Before**: ~2 hours per epoch
- **After**: ~5-10 minutes total ⚡

### 5. **Files Removed**
- Deleted 2.4GB virtual environment folder
- Removed unnecessary documentation files

## To Run:
```bash
cd /home/kushsoni/Desktop/ai-engineer-roadmap-y1/06-The_ML_Core/02_LSTM_Sentiment_Analysis
/home/kushsoni/Desktop/ai-engineer-roadmap-y1/06-The_ML_Core/ai_roadmap_env/bin/python src/train.py
```

The model will train much faster now! 🚀
