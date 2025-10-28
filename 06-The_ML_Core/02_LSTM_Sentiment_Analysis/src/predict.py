import torch
import json
import os
from nltk.tokenize import word_tokenize
import numpy as np
from src.model import SentimentLSTM

class SentimentPredictor:
    """
    A class for loading the trained LSTM model and making predictions on new reviews.
    """
    def __init__(self, model_path='models/lstm_sentiment_model.pth', vocab_path='data/word2idx.json'):
        """
        Initialize the predictor by loading the model and vocabulary.
        
        Args:
            model_path (str): Path to the saved model checkpoint.
            vocab_path (str): Path to the vocabulary file.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load vocabulary
        print(f"Loading vocabulary from {vocab_path}...")
        with open(vocab_path, 'r') as f:
            self.word2idx = json.load(f)
        
        # Load model checkpoint
        print(f"Loading model from {model_path}...")
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Extract hyperparameters
        vocab_size = checkpoint['vocab_size']
        output_size = checkpoint['output_size']
        embedding_dim = checkpoint['embedding_dim']
        hidden_dim = checkpoint['hidden_dim']
        n_layers = checkpoint['n_layers']
        self.seq_length = checkpoint['seq_length']
        
        # Initialize model
        self.model = SentimentLSTM(vocab_size, output_size, embedding_dim, hidden_dim, n_layers)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print("✅ Model loaded successfully!")
        print(f"   Device: {self.device}")
        print(f"   Vocabulary size: {vocab_size}")
        print(f"   Sequence length: {self.seq_length}")
    
    def preprocess_text(self, text):
        """
        Preprocess a single text review for prediction.
        
        Args:
            text (str): The input review text.
            
        Returns:
            torch.Tensor: The preprocessed and padded sequence.
        """
        # Tokenize
        tokens = word_tokenize(text.lower())
        
        # Convert to indices
        numerical = [self.word2idx.get(token, self.word2idx['<unk>']) for token in tokens]
        
        # Pad sequence
        padded = np.zeros(self.seq_length, dtype=int)
        if len(numerical) < self.seq_length:
            padded[:len(numerical)] = numerical
        else:
            padded[:] = numerical[:self.seq_length]
        
        # Convert to tensor
        return torch.from_numpy(padded).unsqueeze(0).to(self.device)
    
    def predict(self, text, return_prob=False):
        """
        Predict the sentiment of a given text.
        
        Args:
            text (str): The input review text.
            return_prob (bool): If True, return the probability as well.
            
        Returns:
            str or tuple: The predicted sentiment ('Positive' or 'Negative'),
                         optionally with the probability.
        """
        # Preprocess
        input_tensor = self.preprocess_text(text)
        
        # Initialize hidden state
        h = self.model.init_hidden(1)
        
        # Make prediction
        with torch.no_grad():
            output, _ = self.model(input_tensor, h)
            prob = output.item()
            prediction = "Positive" if prob > 0.5 else "Negative"
        
        if return_prob:
            return prediction, prob
        else:
            return prediction
    
    def predict_batch(self, texts):
        """
        Predict sentiments for multiple texts.
        
        Args:
            texts (list): A list of review texts.
            
        Returns:
            list: A list of tuples containing (text, prediction, probability).
        """
        results = []
        for text in texts:
            prediction, prob = self.predict(text, return_prob=True)
            results.append((text, prediction, prob))
        return results


def main():
    """
    Main function to demonstrate the sentiment predictor.
    """
    print("\n" + "="*70)
    print("🎬 LSTM Sentiment Analysis - Prediction System")
    print("="*70 + "\n")
    
    # Initialize predictor
    try:
        predictor = SentimentPredictor()
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("\nPlease ensure you have:")
        print("  1. Trained the model using 'python src/train.py'")
        print("  2. Model file exists at 'models/lstm_sentiment_model.pth'")
        print("  3. Vocabulary file exists at 'data/word2idx.json'")
        return
    
    print("\n" + "-"*70)
    
    # Test examples
    test_reviews = [
        "The movie was utterly boring and a waste of time.",
        "This film is absolutely brilliant! A masterpiece of cinema.",
        "I loved every minute of it. The acting was superb and the story was captivating.",
        "Terrible movie. Poor acting and weak storyline. Very disappointed.",
        "An okay film, nothing special but not bad either.",
        "One of the best movies I've ever seen! Highly recommended!",
        "I fell asleep halfway through. Not engaging at all.",
        "Amazing cinematography and outstanding performances from the cast.",
    ]
    
    print("\n📊 Predicting sentiments for sample reviews:\n")
    
    for i, review in enumerate(test_reviews, 1):
        prediction, prob = predictor.predict(review, return_prob=True)
        
        # Display with emoji
        emoji = "😊" if prediction == "Positive" else "😞"
        
        print(f"Review {i}:")
        print(f"  Text: \"{review}\"")
        print(f"  Prediction: {emoji} {prediction} (confidence: {prob:.2%})")
        print()
    
    print("-"*70)
    
    # Interactive mode
    print("\n💬 Interactive Mode - Enter your own reviews (type 'quit' to exit):\n")
    
    while True:
        user_input = input("Enter a movie review: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Thank you for using the sentiment analyzer!")
            break
        
        if not user_input:
            print("Please enter a valid review.\n")
            continue
        
        prediction, prob = predictor.predict(user_input, return_prob=True)
        emoji = "😊" if prediction == "Positive" else "😞"
        
        print(f"\n  Prediction: {emoji} {prediction} (confidence: {prob:.2%})\n")
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()
