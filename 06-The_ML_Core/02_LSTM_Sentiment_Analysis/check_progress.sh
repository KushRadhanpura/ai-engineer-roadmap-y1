#!/bin/bash
echo "=== Training Progress Check ==="
echo ""
echo "Checking if training is running..."
ps aux | grep train.py | grep -v grep && echo "✓ Training is running!" || echo "✗ Training completed or not running"
echo ""
echo "Checking for saved model..."
if [ -f "models/lstm_sentiment_model.pth" ]; then
    echo "✓ Model saved! Training completed successfully!"
    ls -lh models/lstm_sentiment_model.pth
else
    echo "⏳ Model not yet saved - training still in progress..."
fi
echo ""
echo "Recent output from data directory:"
ls -lht data/ 2>/dev/null | head -5
