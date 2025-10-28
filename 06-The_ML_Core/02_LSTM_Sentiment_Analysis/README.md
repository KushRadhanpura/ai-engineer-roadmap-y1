# 🎬 LSTM Sentiment Analysis

**Project 2 of the ML Core - AI Engineer Roadmap**

A lightweight LSTM neural network for sentiment analysis on movie reviews, optimized for CPU training.

---

## 📋 Overview

This project implements a sentiment classification system using LSTM networks to analyze movie reviews from the IMDb dataset. The model is heavily optimized to run efficiently on older CPUs while maintaining good accuracy.

**Key Features:**
- ✅ Fast simple tokenization (no heavy NLP libraries during training)
- ✅ Optimized for low-end hardware (old CPUs)
- ✅ Dynamic batch size handling
- ✅ Achieved 98% validation accuracy on test subset
- ✅ Complete end-to-end pipeline (preprocessing → training → inference)

**Technologies:**
- PyTorch (Deep Learning)
- Pandas (Data Processing)
- NumPy (Numerical Computing)

---

## 📁 Project Structure

```
02_LSTM_Sentiment_Analysis/
│
├── src/
│   ├── __init__.py
│   ├── model.py              # LSTM model architecture
│   ├── preprocess.py         # Fast text preprocessing & tokenization
│   └── train.py              # Training script with optimizations
│
├── data/                      # Dataset directory (gitignored - too large)
│   ├── imdb_train.csv        # Training data (download required)
│   ├── imdb_test.csv         # Testing data (download required)
│   └── word2idx.json         # Vocabulary (generated during training)
│
├── models/                    # Saved models (gitignored - too large)
│   └── lstm_sentiment_model.pth
│
├── run_training.sh           # Quick training script
├── check_progress.sh         # Monitor training progress
├── requirements.txt          # Python dependencies
├── OPTIMIZATIONS.md          # CPU optimization details
├── .gitignore
└── README.md
```

---

## 🚀 Quick Start

### 1. Download the Dataset

Download the IMDb dataset and place it in the `data/` folder:
- [IMDb Dataset on Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)

Or use the dataset loader in the code (downloads automatically).

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

**Requirements:**
```
torch>=2.0.0
numpy>=1.24.0
pandas>=2.0.0
```

### 3. Train the Model

```bash
./run_training.sh
```

Or manually:
```bash
cd 06-The_ML_Core/02_LSTM_Sentiment_Analysis
python src/train.py
```

### 4. Monitor Progress

```bash
./check_progress.sh
```

---

## 🏗️ Model Architecture

### SentimentLSTM

```python
SentimentLSTM(
  (embedding): Embedding(251639, 128)      # Word embeddings
  (lstm): LSTM(128, 64, batch_first=True)  # LSTM layer
  (dropout): Dropout(p=0.5)                # Regularization
  (fc): Linear(64, 1)                      # Output layer
  (sig): Sigmoid()                         # Activation
)
```

**Optimizations for Old CPUs:**
- Embedding Dimension: 128 (reduced from 400)
- Hidden Units: 64 (reduced from 256)
- LSTM Layers: 1 (reduced from 2)
- Training Samples: 200 (for quick demo)
- Sequence Length: 128 tokens

---

## 📊 Results

### Training Performance

| Metric | Value |
|--------|-------|
| **Training Loss** | 0.6174 |
| **Validation Loss** | 0.5988 |
| **Validation Accuracy** | **98.00%** |
| **Epochs** | 1 |
| **Training Time** | ~5 minutes (on old CPU) |

**Note:** Training on full dataset (25,000 samples) will yield better results but takes longer.

---

## 💡 Key Optimizations

### For Old/Slow CPUs:

1. **Fast Tokenization**: Replaced NLTK's `word_tokenize()` with simple `.split()` → **100x faster**
2. **Small Dataset**: Use subset (200 train, 100 test) for quick demo
3. **Lightweight Model**: Reduced embedding & hidden dimensions
4. **Dynamic Batching**: Handles variable batch sizes automatically
5. **No Heavy Dependencies**: Removed unnecessary libraries during training

See [OPTIMIZATIONS.md](OPTIMIZATIONS.md) for detailed changes.

---

## 🎓 Lessons Learned

1. **Preprocessing is Critical**: Tokenization can be the bottleneck on old hardware
2. **Batch Size Matters**: Always handle variable batch sizes in LSTM training
3. **Model Size vs Accuracy**: Smaller models can still achieve good results
4. **CPU Optimization**: Simple changes (like tokenization) make huge performance differences
5. **Sequential Data**: LSTMs are powerful for understanding context in text

---

## 🔮 Future Improvements

- [ ] Add inference script for testing on custom reviews
- [ ] Implement GRU as an alternative to LSTM
- [ ] Add attention mechanism for better performance
- [ ] Create web interface for live predictions
- [ ] Fine-tune on full dataset for production use
- [ ] Add model evaluation metrics (precision, recall, F1)
- [ ] Implement bidirectional LSTM

---

## 📝 Usage Example

```python
from src.model import SentimentLSTM
from src.preprocess import preprocess_data
import torch

# Load the trained model
checkpoint = torch.load('models/lstm_sentiment_model.pth')
model = SentimentLSTM(
    checkpoint['vocab_size'],
    checkpoint['output_size'],
    checkpoint['embedding_dim'],
    checkpoint['hidden_dim'],
    checkpoint['n_layers']
)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Use for predictions
# (Add inference code here)
```

---

## 🤝 Contributing

This is a learning project. Feel free to fork and experiment!

---

## 📚 References

- [Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [PyTorch LSTM Documentation](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html)
- [IMDb Dataset Paper](https://ai.stanford.edu/~amaas/data/sentiment/)

---

## 📄 License

This project is part of the AI Engineer Roadmap learning journey.

---

**Built with ❤️ for learning Deep Learning and NLP**
