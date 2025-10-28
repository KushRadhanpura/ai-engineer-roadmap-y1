import torch
import torch.nn as nn

class SentimentLSTM(nn.Module):
    """
    An LSTM model for sentiment analysis.
    """
    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, drop_prob=0.5):
        """
        Initialize the model by setting up the layers.

        Args:
            vocab_size (int): The size of the vocabulary.
            output_size (int): The size of the output layer (e.g., 1 for binary classification).
            embedding_dim (int): The dimensionality of the word embeddings.
            hidden_dim (int): The number of features in the hidden state of the LSTM.
            n_layers (int): The number of recurrent layers.
            drop_prob (float): The dropout probability.
        """
        super(SentimentLSTM, self).__init__()

        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        # --- Layers ---
        # 1. Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # 2. LSTM layer
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, 
                            dropout=drop_prob, batch_first=True)
        
        # 3. Dropout layer
        self.dropout = nn.Dropout(drop_prob)
        
        # 4. Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_size)
        
        # 5. Sigmoid activation function
        self.sig = nn.Sigmoid()

    def forward(self, x, hidden):
        """
        Perform a forward pass of our model on some input and hidden state.

        Args:
            x (torch.Tensor): The input tensor (batch of reviews).
            hidden (tuple): The initial hidden and cell states.

        Returns:
            tuple: A tuple containing the output of the sigmoid function and the
                   final hidden state.
        """
        batch_size = x.size(0)

        # 1. Embeddings and LSTM output
        embeds = self.embedding(x)
        lstm_out, hidden = self.lstm(embeds, hidden)
        
        # 2. Stack up lstm outputs
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)
        
        # 3. Dropout and fully-connected layer
        out = self.dropout(lstm_out)
        out = self.fc(out)
        
        # 4. Sigmoid function
        sig_out = self.sig(out)
        
        # 5. Reshape to be batch_size first
        sig_out = sig_out.view(batch_size, -1)
        sig_out = sig_out[:, -1] # get last batch of labels
        
        # return last sigmoid output and hidden state
        return sig_out, hidden

    def init_hidden(self, batch_size):
        """
        Initializes hidden state.
        
        Create two new tensors with sizes n_layers x batch_size x hidden_dim,
        initialized to zero, for hidden state and cell state of LSTM.
        
        Args:
            batch_size (int): The size of the batch.

        Returns:
            tuple: A tuple of two zeroed tensors for the hidden and cell states.
        """
        # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data
        
        if (torch.cuda.is_available()):
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda(),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())
        
        return hidden
