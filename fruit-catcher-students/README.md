# Fruit Catcher AI Project

This project implements a game where players need to catch fruits while avoiding bombs. The game includes both human playable mode and AI-powered classifiers to make decisions.

## Project Structure

- `main.py`: Main game runner and dataset loader
- `dt.py`: Decision Tree implementation for fruit/bomb classification
- `game.py`: Game mechanics and visualization
- `nn.py`: Neural Network implementation
- `genetic.py`: Genetic Algorithm implementation
- `train.csv`: Training dataset
- `test.csv`: Test dataset
- `items.csv`: Game items configuration

## Decision Tree Implementation (dt.py)

The decision tree classifier is implemented in `dt.py` with the following key components:

### Core Functions

1. `train_decision_tree(X, y, feature_names)`
   - Main entry point for training
   - Takes feature vectors, labels, and feature names
   - Returns a trained DecisionTree instance

2. `build_decision_tree(X, y, feature_names, depth, max_depth)`
   - Recursive tree building function
   - Uses information gain to choose best splits
   - Implements stopping criteria (max depth, pure nodes)

3. `calculate_entropy(y)`
   - Calculates Shannon entropy for label distributions
   - Used for information gain calculations

4. `calculate_feature_entropy(features, feature_idx, labels)`
   - Computes entropy for specific feature splits
   - Helps determine the best feature to split on

5. `calculate_information_gain(feature_entropies, dataset_entropy)`
   - Calculates information gain for all features
   - Used to select the best splitting feature

6. `print_tree(tree, indent="")`
   - Visualizes the decision tree structure
   - Shows feature splits and classifications

### DecisionTree Class

```python
class DecisionTree:
    def __init__(self, X, y, threshold=1.0, max_depth=None):
        # Initialize tree parameters
        
    def predict(self, x):
        # Make predictions on new data
```

## Neural Network Implementation (nn.py)

The neural network controller is implemented in `nn.py` and is responsible for controlling the basket's movement in the game.

### Core Components

1. `NeuralNetwork` Class
   - Implements a feedforward neural network
   - Takes game state as input (basket position + falling items)
   - Outputs movement decisions (-1 for left, 1 for right)
   - Configurable architecture with multiple hidden layers

2. `create_network_architecture(input_size)`
   - Creates a network with predefined architecture:
     - 2 hidden layers (8 and 4 neurons)
     - Sigmoid activation for hidden layers
     - Step function for output layer

### Network Architecture

The neural network processes:
- Input Layer (10 neurons):
  - Basket position (1 value)
  - Up to 3 falling items (3 values each):
    - x coordinate
    - y coordinate
    - is_fruit prediction (-1 or 1)
- Hidden Layers:
  - First hidden layer: 8 neurons
  - Second hidden layer: 4 neurons
- Output Layer:
  - Single neuron
  - Step function activation (-1 or 1)

### Training

The network is trained using a genetic algorithm that:
1. Generates a population of random weights
2. Evaluates each network by playing the game
3. Selects best performers for next generation
4. Applies crossover and mutation
5. Repeats until target score is reached

### Usage

```python
# Create and train the network
python main.py -t -p 100 -g 100  # Train with population 100, generations 100

# Play the game with trained network
python main.py  # Shows game interface with AI option
```

## Integration with main.py

The main.py file was modified to properly pass feature names to the decision tree:

```python
def train_fruit_classifier(filename):
    feature_names, X, y = load_train_dataset(filename)
    dt = train_decision_tree(X, y, feature_names=feature_names)
    return lambda item: dt.predict(item)
```

## Data Format

The training data (`train.csv`) follows this structure:
- Column 1: ID
- Columns 2-4: Features (name, color, format)
- Last Column: Label (1 for fruit, -1 for bomb)

## Decision Tree Features

1. **Automatic Feature Detection**
   - Features are automatically extracted from CSV headers
   - No hardcoded feature names required

2. **Information Gain Based Splitting**
   - Uses entropy calculations to find optimal splits
   - Maximizes information gain at each node

3. **Tree Visualization**
   - Includes a pretty-print function for the tree
   - Shows the complete decision path

4. **Configurable Parameters**
   - Adjustable maximum depth
   - Customizable splitting threshold

## Usage

To run the game with the decision tree classifier:
```bash
python main.py
```

To run in headless mode (no graphics):
```bash
python main.py --headless
```
