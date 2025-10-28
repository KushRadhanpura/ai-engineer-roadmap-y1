import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from src.preprocess import preprocess_data
from src.model import SentimentLSTM
import os
import json

# --- Configuration ---
DATA_DIR = 'data'
TRAIN_CSV = os.path.join(DATA_DIR, 'imdb_train.csv')
TEST_CSV = os.path.join(DATA_DIR, 'imdb_test.csv')
MODEL_DIR = 'models'
os.makedirs(MODEL_DIR, exist_ok=True)

# Hyperparameters
VOCAB_SIZE = 0  # Will be set after preprocessing
OUTPUT_SIZE = 1
EMBEDDING_DIM = 400
HIDDEN_DIM = 256
N_LAYERS = 2
LR = 0.001
EPOCHS = 4
BATCH_SIZE = 50
SEQ_LENGTH = 256

# --- 1. Preprocess Data ---
print("Preprocessing training data...")
train_data, train_labels, word2idx = preprocess_data(
    TRAIN_CSV, DATA_DIR, is_train=True, seq_length=SEQ_LENGTH
)
VOCAB_SIZE = len(word2idx)

print("Preprocessing test data...")
test_data, test_labels = preprocess_data(
    TEST_CSV, DATA_DIR, is_train=False, word2idx=word2idx, seq_length=SEQ_LENGTH
)

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
model.to(device)

for epoch in range(EPOCHS):
    model.train()
    h = model.init_hidden(BATCH_SIZE)
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Creating new variables for the hidden state, otherwise
        # we'd backprop through the entire training history
        h = tuple([each.data for each in h])

        model.zero_grad()
        output, h = model(inputs, h)
        
        loss = criterion(output.squeeze(), labels.float())
        loss.backward()
        optimizer.step()

    # --- Validation ---
    model.eval()
    val_h = model.init_hidden(BATCH_SIZE)
    val_losses = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            val_h = tuple([each.data for each in val_h])
            
            output, val_h = model(inputs, val_h)
            val_loss = criterion(output.squeeze(), labels.float())
            val_losses.append(val_loss.item())
            
    print(f"Epoch {epoch+1}/{EPOCHS}.. "
          f"Train Loss: {loss.item():.3f}.. "
          f"Val Loss: {torch.mean(torch.tensor(val_losses)):.3f}")

# --- 6. Save the Model ---
model_path = os.path.join(MODEL_DIR, 'sentiment_lstm.pth')
torch.save(model.state_dict(), model_path)
print(f"\nModel saved to {model_path}")
