import numpy as np

def calculate_hidden_size(input_size, output_size=1):
    """Calculate a good number of hidden neurons using various rules of thumb.
    
    Common rules of thumb:
    1. Mean of input and output size
    2. 2/3 of input size + output size
    3. Square root of input size times output size
    
    Args:
        input_size: Number of input features
        output_size: Number of outputs (default 1 for our case)
        
    Returns:
        int: Recommended number of hidden neurons
    """
    rules = [
        # Mean of input and output
        (input_size + output_size) // 2,
        
        # 2/3 of input layer + output size
        (2 * input_size) // 3 + output_size,
        
        # Square root of input * output
        int(np.sqrt(input_size * output_size)) * 2
    ]
    
    # Take the average of all rules and ensure it's at least 4 neurons
    hidden_size = max(4, int(np.mean(rules)))
    
    return hidden_size

def relu(x):
    """Simple ReLU activation function."""
    return np.maximum(0, x)

def sigmoid(x):
    """Sigmoid activation function."""
    return 1 / (1 + np.exp(-x))

class NeuralNetwork:
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = 1
        
    def compute_num_weights(self):
        """Calculate total number of weights and biases needed."""
        # Weights and biases for hidden layer
        w1 = self.input_size * self.hidden_size
        b1 = self.hidden_size
        
        # Weights and bias for output layer
        w2 = self.hidden_size * self.output_size
        b2 = self.output_size
        
        return w1 + w2 + b1 + b2
        
    def load_weights(self, weights):
        """Load weights and biases into the network."""
        weights = np.array(weights)
        idx = 0
        
        # Hidden layer
        w1_size = self.input_size * self.hidden_size
        self.hidden_weights = weights[idx:idx+w1_size].reshape(self.input_size, self.hidden_size)
        idx += w1_size
        
        self.hidden_biases = weights[idx:idx+self.hidden_size]
        idx += self.hidden_size
        
        # Output layer
        w2_size = self.hidden_size
        self.output_weights = weights[idx:idx+w2_size]
        idx += w2_size
        
        self.output_bias = weights[idx]
        
    def forward(self, x):
        """Forward pass through the network."""
        # Hidden layer with ReLU
        hidden = relu(np.dot(x, self.hidden_weights) + self.hidden_biases)
        
        # Output layer
        output = np.dot(hidden, self.output_weights) + self.output_bias
        
        # Simple decision making
        return 1 if output > 0 else -1

def create_network_architecture(input_size):
    """Create a simple neural network with one hidden layer."""
    hidden_size = calculate_hidden_size(input_size) # Fixed size hidden layer
    return NeuralNetwork(input_size, hidden_size)
