import pandas as pd
# import nltk  # Commented out - too slow for old CPU
# from nltk.tokenize import word_tokenize
from collections import Counter
import numpy as np
import os
import json

def simple_tokenize(text):
    """Fast simple tokenization - much faster than NLTK for old CPUs"""
    return text.lower().split()

def preprocess_data(csv_path, data_dir, is_train=True, word2idx=None, seq_length=256):
    """
    Preprocesses the IMDb dataset.

    Args:
        csv_path (str): Path to the input CSV file.
        data_dir (str): Directory to save processed data and vocabulary.
        is_train (bool): Whether this is the training set. If True, a new
                         vocabulary will be created.
        word2idx (dict): A pre-existing word-to-index mapping. Required if
                         is_train is False.
        seq_length (int): The fixed sequence length for padding.

    Returns:
        tuple: A tuple containing padded sequences and labels. If is_train
               is True, also returns the word2idx mapping.
    """
    # Load the data
    df = pd.read_csv(csv_path)
    df['text'] = df['text'].astype(str)

    # --- 1. Tokenization (FAST SIMPLE METHOD for old CPU) ---
    print("Tokenizing text...")
    df['tokens'] = df['text'].apply(simple_tokenize)

    # --- 2. Vocabulary Building (only for training set) ---
    if is_train:
        all_tokens = [token for tokens_list in df['tokens'] for token in tokens_list]
        word_counts = Counter(all_tokens)
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        
        word2idx = {'<pad>': 0, '<unk>': 1}
        for i, (word, count) in enumerate(sorted_words):
            word2idx[word] = i + 2
            
        # Save the vocabulary
        vocab_path = os.path.join(data_dir, 'word2idx.json')
        with open(vocab_path, 'w') as f:
            json.dump(word2idx, f)

    if word2idx is None:
        raise ValueError("word2idx must be provided for the test set.")

    # --- 3. Numericalization ---
    df['numerical'] = df['tokens'].apply(
        lambda tokens: [word2idx.get(token, word2idx['<unk>']) for token in tokens]
    )

    # --- 4. Padding ---
    padded_sequences = np.zeros((len(df), seq_length), dtype=int)
    for i, numerical_list in enumerate(df['numerical']):
        if len(numerical_list) < seq_length:
            padded_sequences[i, :len(numerical_list)] = numerical_list
        else:
            padded_sequences[i, :] = numerical_list[:seq_length]

    labels = df['label'].to_numpy()

    if is_train:
        return padded_sequences, labels, word2idx
    else:
        return padded_sequences, labels
