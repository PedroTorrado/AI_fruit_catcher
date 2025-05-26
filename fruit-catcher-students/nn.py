import numpy as np
import random

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
        (input_size + output_size) // 2,
        (2 * input_size) // 3 + output_size,
        int(np.sqrt(input_size * output_size)) * 2
    ]
    return max(4, int(np.mean(rules)))


def relu(x):
    return np.maximum(0, x)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class NeuralNetwork:
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = 1

    def compute_num_weights(self):
        w1 = self.input_size * self.hidden_size
        b1 = self.hidden_size
        w2 = self.hidden_size * self.output_size
        b2 = self.output_size
        return w1 + w2 + b1 + b2

    def load_weights(self, weights):
        weights = np.array(weights)
        idx = 0
        w1_size = self.input_size * self.hidden_size
        self.hidden_weights = weights[idx:idx + w1_size].reshape(self.input_size, self.hidden_size)
        idx += w1_size
        self.hidden_biases = weights[idx:idx + self.hidden_size]
        idx += self.hidden_size
        w2_size = self.hidden_size * self.output_size
        self.output_weights = weights[idx:idx + w2_size].reshape(self.hidden_size, self.output_size)
        idx += w2_size
        self.output_bias = weights[idx]

    def preprocess(self, state):
        """Apply domain-specific transformations to the input state."""
        # Normalize basket x-pos [-1, 1]
        state[0] = (state[0] - 0.5) * 2
        # Compute relative positions for each item to the basket
        basket_x = state[0] / 2 + 0.5  # Convert back to [0,1] for comparison
        for i in range(3):
            base = 1 + i * 3
            if base + 2 >= len(state):
                continue
            fruit_x = state[base]
            fruit_y = state[base + 1]
            is_fruit = state[base + 2]
            # Compute relative x position to basket
            rel_x = fruit_x - basket_x
            # Emphasize fruit/bomb status (bombs negative, fruit positive)
            state[base + 2] = is_fruit * 3.0  # Moderate emphasis on fruit/bomb status
            # Store relative x position with stronger bias for bombs
            if is_fruit < 0:  # If it's a bomb
                # Amplify the relative position to make the basket move away more strongly
                state[base] = rel_x * 4.0  # Strong emphasis on bomb avoidance
            else:
                state[base] = rel_x * 2.0  # Normal emphasis for fruits
            # Emphasize vertical position as urgency
            state[base + 1] = fruit_y * 1.5  # Moderate emphasis on vertical urgency
        return state

    def forward(self, x):
        """Forward pass with state preprocessing."""
        x = self.preprocess(x)
        hidden = relu(np.dot(x, self.hidden_weights) + self.hidden_biases)
        output = sigmoid(np.dot(hidden, self.output_weights) + self.output_bias)
        return 1 if output > 0.5 else -1


def create_network_architecture(input_size):
    hidden_size = calculate_hidden_size(input_size)
    return NeuralNetwork(input_size, hidden_size)
