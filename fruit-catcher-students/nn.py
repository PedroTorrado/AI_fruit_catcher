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
        """
        Preprocess the game state to help the neural network make better decisions.
        
        The state contains:
        - state[0]: Basket x-position (normalized to [-1,1])
        - For each item (up to 3 items):
            - X position 
            - Y position
            - Type (1 for fruit, -1 for bomb)
        """
        # Normalize basket position from [0,1] to [-1,1] range
        basket_pos = (state[0] - 0.5) * 2
        state[0] = basket_pos
        
        # Get original basket x-pos in [0,1] range for distance calculations
        basket_x = (basket_pos + 1) / 2
        
        # Process each item (fruit/bomb)
        for i in range(3):
            # Get start index for this item's data
            idx = 1 + i * 3
            
            # Skip if beyond state bounds
            if idx + 2 >= len(state):
                continue
                
            # Get item properties
            item_x = state[idx]
            item_y = state[idx + 1] 
            is_fruit = state[idx + 2]
            
            # Calculate horizontal distance to basket
            distance_to_basket = item_x - basket_x
            
            # Update state values:
            # 1. Horizontal distance (weighted more for closer items)
            closeness_weight = 1.0 + (1.0 - item_y)  # Items closer to ground weighted more
            state[idx] = distance_to_basket * 2.0 * closeness_weight
            
            # 2. Vertical position (higher value = more urgent)
            state[idx + 1] = item_y * 3.0
            
            # 3. Item type (amplify fruit/bomb difference)
            state[idx + 2] = is_fruit * 3.0
            
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
