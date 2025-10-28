# 🧠 The ML Core

**Advanced Machine Learning & Deep Learning Projects**

This section contains fundamental machine learning and deep learning projects that form the core of AI engineering skills.

---

## 📚 Projects

### 01. Transfer Learning Classifier 🖼️
**Rock, Paper, Scissors Image Classification**

A transfer learning project using pre-trained ResNet50 to classify images of rock, paper, and scissors.

**Key Concepts:**
- Transfer Learning with ResNet50
- PyTorch Image Classification
- Data Augmentation
- Fine-tuning Pre-trained Models

**Technologies:** PyTorch, torchvision, PIL

📁 [View Project](./01_TransferLearning_Classifier/)

---

### 02. LSTM Sentiment Analysis 🎬
**Movie Review Sentiment Classification**

An LSTM-based sentiment analysis system for classifying movie reviews, optimized for CPU training.

**Key Concepts:**
- Recurrent Neural Networks (LSTM)
- Natural Language Processing
- Text Preprocessing & Tokenization
- Sequential Data Processing
- CPU Optimization Techniques

**Technologies:** PyTorch, Pandas, NumPy

**Results:** 98% validation accuracy

📁 [View Project](./02_LSTM_Sentiment_Analysis/)

---

## 🎯 Learning Objectives

Through these projects, you will learn:

1. **Transfer Learning**
   - Using pre-trained models
   - Fine-tuning for specific tasks
   - Efficient training with limited data

2. **Deep Learning Fundamentals**
   - Neural network architectures (CNN, RNN/LSTM)
   - Backpropagation and gradient descent
   - Loss functions and optimizers

3. **Natural Language Processing**
   - Text tokenization and preprocessing
   - Word embeddings
   - Sequential data modeling

4. **Model Optimization**
   - Training on limited hardware
   - Batch size handling
   - Performance optimization techniques

5. **PyTorch Ecosystem**
   - Building custom models
   - Data loaders and transforms
   - Model checkpointing and evaluation

---

## 🛠️ Setup

### Prerequisites

```bash
Python 3.8+
PyTorch 2.0+
```

### Installation

Each project has its own `requirements.txt`. Navigate to the project directory and install:

```bash
cd 01_TransferLearning_Classifier  # or 02_LSTM_Sentiment_Analysis
pip install -r requirements.txt
```

---

## 📊 Project Comparison

| Project | Task Type | Model | Dataset | Difficulty |
|---------|-----------|-------|---------|------------|
| **01 - Transfer Learning** | Image Classification | ResNet50 | Rock/Paper/Scissors | ⭐⭐ |
| **02 - LSTM Sentiment** | Text Classification | LSTM | IMDb Reviews | ⭐⭐⭐ |

---

## 🚀 Getting Started

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd 06-The_ML_Core
   ```

2. **Choose a project**
   ```bash
   cd 01_TransferLearning_Classifier
   # or
   cd 02_LSTM_Sentiment_Analysis
   ```

3. **Follow the project README**
   - Each project has detailed setup and usage instructions

---

## 📈 Progress Tracking

- [x] 01 - Transfer Learning Classifier (Completed)
- [x] 02 - LSTM Sentiment Analysis (Completed)
- [ ] More projects coming soon...

---

## 🎓 Key Takeaways

### From Transfer Learning Project:
- Pre-trained models provide excellent starting points
- Data augmentation is crucial for small datasets
- Fine-tuning requires careful hyperparameter selection

### From LSTM Sentiment Analysis:
- Preprocessing can be the biggest bottleneck
- Smaller models can still achieve great results
- Hardware optimization is essential for practical deployment
- Dynamic batch handling prevents runtime errors

---

## 🔗 Resources

### Learning Materials:
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [Understanding LSTMs](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [Transfer Learning Guide](https://cs231n.github.io/transfer-learning/)

### Datasets:
- [ImageNet](https://www.image-net.org/)
- [IMDb Dataset](https://ai.stanford.edu/~amaas/data/sentiment/)

---

## 🤝 Contributing

These are learning projects. Feel free to:
- Fork and experiment
- Try different architectures
- Optimize for your hardware
- Share your improvements

---

## 📝 Notes

- All projects are optimized for learning, not production
- Models are trained on subsets for quick iteration
- Full training on complete datasets will yield better results
- Hardware requirements vary by project

---

## 📄 License

Part of the AI Engineer Roadmap learning journey.

---

**Happy Learning! 🚀**

*Building AI engineering skills one project at a time.*
