# simple_nn.py

import numpy as np

# --- 1. Activation Functions ---

def sigmoid(x):
    """Sigmoid activation function."""
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    """Derivative of the sigmoid function."""
    s = sigmoid(x)
    return s * (1 - s)

# --- 2. Neural Network Class ---

class SimpleNeuralNetwork:
    """A two-layer neural network implemented from scratch to solve XOR."""

    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights with small random values (Heavier initialization can help XOR)
        np.random.seed(1) # Ensures reproducible results
        
        # W1: (input_size x hidden_size)
        self.W1 = 2 * np.random.random((input_size, hidden_size)) - 1
        self.b1 = np.zeros((1, hidden_size))
        
        # W2: (hidden_size x output_size)
        self.W2 = 2 * np.random.random((hidden_size, output_size)) - 1
        self.b2 = np.zeros((1, output_size))

    def forward(self, X):
        """Performs the forward pass."""
        # Layer 1: Input -> Hidden
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = sigmoid(self.Z1)
        
        # Layer 2: Hidden -> Output
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = sigmoid(self.Z2) # Final Output (Prediction)
        
        return self.A2

    def train(self, X, Y, epochs=10000, learning_rate=0.1):
        """Trains the network using backpropagation."""
        
        for epoch in range(1, epochs + 1):
            
            # 1. Forward Propagation
            self.A2 = self.forward(X)
            
            # 2. Calculate Loss (Mean Squared Error)
            loss = np.mean((Y - self.A2) ** 2)
            
            # 3. Backward Propagation
            
            # Error at the Output Layer
            E2 = Y - self.A2
            
            # Delta at the Output Layer
            dZ2 = E2 * sigmoid_derivative(self.Z2) # Element-wise multiplication
            
            # Error at the Hidden Layer
            E1 = dZ2.dot(self.W2.T)
            
            # Delta at the Hidden Layer
            dZ1 = E1 * sigmoid_derivative(self.Z1)
            
            # 4. Update Weights and Biases (Gradient Descent)
            
            # Calculate gradients
            dW2 = self.A1.T.dot(dZ2)
            db2 = np.sum(dZ2, axis=0, keepdims=True)
            
            dW1 = X.T.dot(dZ1)
            db1 = np.sum(dZ1, axis=0, keepdims=True)
            
            # Update parameters
            self.W2 += dW2 * learning_rate
            self.b2 += db2 * learning_rate
            self.W1 += dW1 * learning_rate
            self.b1 += db1 * learning_rate
            
            # Print status every 1000 epochs
            if epoch % 1000 == 0 or epoch == 1:
                print(f"Epoch {epoch:5d}, Loss: {loss:.4f}")

# --- 3. Training and Testing Data (XOR) ---

if __name__ == "__main__":
    
    # Input Data for XOR
    X_train = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])
    
    # Expected Output Data for XOR
    Y_train = np.array([[0], [1], [1], [0]])
    
    # --- 4. Create and Train the Network ---
    
    input_dim = 2
    hidden_dim = 4 # Use at least 2 hidden neurons for XOR, 4 is a common choice
    output_dim = 1
    
    nn = SimpleNeuralNetwork(input_dim, hidden_dim, output_dim)
    
    print("\n--- Training the Simple Neural Network ---")
    nn.train(X_train, Y_train, epochs=10000, learning_rate=0.1)
    
    # --- 5. Test the Trained Network ---
    
    print("\n--- Testing the Trained Network ---")
    
    # Run the trained network on the input data
    predictions = nn.forward(X_train)
    
    # Display results
    for i in range(len(X_train)):
        input_data = X_train[i]
        true_value = Y_train[i][0]
        prediction = predictions[i][0]
        
        # Determine the prediction result (round to 0 or 1)
        result = "Success" if np.round(prediction) == true_value else "Fail"
        
        print(f"Input: {input_data} -> True: {true_value:.0f}, Prediction: {prediction:.4f} ({result})")