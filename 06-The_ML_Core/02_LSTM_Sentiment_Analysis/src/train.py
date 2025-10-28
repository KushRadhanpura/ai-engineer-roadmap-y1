import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocess import preprocess_data
from src.model import SentimentLSTM
import json

def train_model():
    """
    Train the LSTM sentiment analysis model and save it to disk.
    """
    # --- Configuration ---
    DATA_DIR = 'data'
    TRAIN_CSV = os.path.join(DATA_DIR, 'imdb_train.csv')
    TEST_CSV = os.path.join(DATA_DIR, 'imdb_test.csv')
    MODEL_DIR = 'models'
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Hyperparameters (Optimized for old CPU)
    VOCAB_SIZE = 0  # Will be set after preprocessing
    OUTPUT_SIZE = 1
    EMBEDDING_DIM = 128  # Reduced from 400
    HIDDEN_DIM = 64      # Reduced from 256
    N_LAYERS = 1         # Reduced from 2
    LR = 0.001
    EPOCHS = 1           # Reduced from 4 (just to see output)
    BATCH_SIZE = 128     # Increased for faster processing
    SEQ_LENGTH = 128     # Reduced from 256 (shorter sequences)

    # --- 1. Preprocess Data ---
    print("Preprocessing training data...")
    train_data, train_labels, word2idx = preprocess_data(
        TRAIN_CSV, DATA_DIR, is_train=True, seq_length=SEQ_LENGTH
    )
    
    # Use only a tiny subset for quick testing on old CPU
    TRAIN_SUBSET = 200  # Reduced from 1000 to 200 samples
    train_data = train_data[:TRAIN_SUBSET]
    train_labels = train_labels[:TRAIN_SUBSET]
    print(f"Using {TRAIN_SUBSET} training samples (reduced for old CPU)")
    
    VOCAB_SIZE = len(word2idx)
    print(f"Vocabulary size: {VOCAB_SIZE}")

    print("Preprocessing test data...")
    test_data, test_labels = preprocess_data(
        TEST_CSV, DATA_DIR, is_train=False, word2idx=word2idx, seq_length=SEQ_LENGTH
    )
    
    # Use only a tiny subset for quick testing
    TEST_SUBSET = 100  # Reduced from 500 to 100 samples
    test_data = test_data[:TEST_SUBSET]
    test_labels = test_labels[:TEST_SUBSET]
    print(f"Using {TEST_SUBSET} test samples (reduced for old CPU)")

    # --- 2. Create DataLoaders ---
    train_dataset = TensorDataset(torch.from_numpy(train_data), torch.from_numpy(train_labels))
    test_dataset = TensorDataset(torch.from_numpy(test_data), torch.from_numpy(test_labels))

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, shuffle=True, batch_size=BATCH_SIZE)

    # --- 3. Instantiate the Model ---
    model = SentimentLSTM(VOCAB_SIZE, OUTPUT_SIZE, EMBEDDING_DIM, HIDDEN_DIM, N_LAYERS)
    print("\nModel architecture:")
    print(model)

    # --- 4. Define Loss and Optimizer ---
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # --- 5. Training Loop ---
    print("\nStarting training...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    model.to(device)

    train_losses = []
    val_losses = []

    for epoch in range(EPOCHS):
        model.train()
        epoch_losses = []
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            # Get actual batch size (handles variable batch sizes)
            current_batch_size = inputs.size(0)
            h = model.init_hidden(current_batch_size)
            
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Creating new variables for the hidden state, otherwise
            # we'd backprop through the entire training history
            h = tuple([each.data for each in h])

            model.zero_grad()
            output, h = model(inputs, h)
            
            loss = criterion(output.squeeze(), labels.float())
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

            if batch_idx % 2 == 0:  # Print every 2 batches (reduced from 100)
                print(f"Epoch {epoch+1}/{EPOCHS}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")

        avg_train_loss = sum(epoch_losses) / len(epoch_losses)
        train_losses.append(avg_train_loss)

        # --- Validation ---
        model.eval()
        batch_val_losses = []
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                # Get actual batch size for validation too
                current_batch_size = inputs.size(0)
                val_h = model.init_hidden(current_batch_size)
                
                inputs, labels = inputs.to(device), labels.to(device)
                val_h = tuple([each.data for each in val_h])
                
                output, val_h = model(inputs, val_h)
                val_loss = criterion(output.squeeze(), labels.float())
                batch_val_losses.append(val_loss.item())
                
                # Calculate accuracy
                preds = (output.squeeze() > 0.5).float()
                correct += (preds == labels.float()).sum().item()
                total += labels.size(0)
        
        avg_val_loss = sum(batch_val_losses) / len(batch_val_losses)
        val_losses.append(avg_val_loss)
        accuracy = 100 * correct / total
                
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{EPOCHS} Summary:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}")
        print(f"  Val Accuracy: {accuracy:.2f}%")
        print(f"{'='*60}\n")

    # --- 6. Save the Model ---
    model_path = os.path.join(MODEL_DIR, 'lstm_sentiment_model.pth')
    
    # Save model state dict along with hyperparameters
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'vocab_size': VOCAB_SIZE,
        'output_size': OUTPUT_SIZE,
        'embedding_dim': EMBEDDING_DIM,
        'hidden_dim': HIDDEN_DIM,
        'n_layers': N_LAYERS,
        'seq_length': SEQ_LENGTH
    }
    
    torch.save(checkpoint, model_path)
    print(f"\n✅ Model saved to {model_path}")
    print(f"✅ Vocabulary saved to {os.path.join(DATA_DIR, 'word2idx.json')}")
    print(f"\nFinal Training Accuracy: {accuracy:.2f}%")
    print(f"Final Validation Loss: {avg_val_loss:.4f}")

if __name__ == "__main__":
    train_model()
