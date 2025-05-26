# Fruit Catcher AI Project

This project implements a game where players need to catch fruits while avoiding bombs. The game includes both human playable mode and AI-powered classifiers to make decisions.

```
    🍎  🍌  🍇
      \  |  /
       \ | /
        \|/
    ┌──────────┐
    │   🧺     │
    └──────────┘
```

## Project Structure

- `main.py`: Main game runner and dataset loader
- `dt.py`: Decision Tree implementation for fruit/bomb classification
- `game.py`: Game mechanics and visualization
- `nn.py`: Neural Network implementation
- `genetic.py`: Genetic Algorithm implementation
- `train.csv`: Training dataset
- `test.csv`: Test dataset
- `items.csv`: Game items configuration
- `weights_logger.py`: Weight logging and analysis system

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

The neural network controller implements a feedforward neural network for game control decisions.

### Network Architecture

```
Input Layer (10)     Hidden Layer 1 (8)     Hidden Layer 2 (4)     Output Layer (1)
    [x]                 [x]                    [x]                    [x]
    [x]                 [x]                    [x]
    [x]                 [x]                    [x]
    [x]                 [x]                    [x]
    [x]                 [x]
    [x]                 [x]
    [x]                 [x]
    [x]                 [x]
    [x]
    [x]
```

### Input Processing
- Basket position (1 value)
- Up to 3 falling items (3 values each):
  - x coordinate
  - y coordinate
  - is_fruit prediction (-1 or 1)

### Layer Details
1. **Input Layer (10 neurons)**
   - Raw game state data
   - Normalized to [-1, 1] range

2. **Hidden Layer 1 (8 neurons)**
   - Sigmoid activation
   - Learns complex patterns in item trajectories

3. **Hidden Layer 2 (4 neurons)**
   - Sigmoid activation
   - Refines decision making

4. **Output Layer (1 neuron)**
   - Step function activation
   - Outputs: -1 (move left) or 1 (move right)

## Genetic Algorithm Implementation (genetic.py)

The genetic algorithm optimizes the neural network weights through evolutionary processes.

### Algorithm Flow
```
Initial Population
      ↓
Fitness Evaluation
      ↓
Selection
      ↓
Crossover
      ↓
Mutation
      ↓
New Generation
```

### Key Components

1. **Population Initialization**
   - Random weight generation
   - Population size configurable
   - Weights normalized to [-1, 1]

2. **Fitness Evaluation**
   - Game simulation for each individual
   - Multiple seeds for robust evaluation
   - Score based on:
     - Fruits caught
     - Bombs avoided
     - Survival time

3. **Selection**
   - Tournament selection
   - Elitism (best individuals preserved)
   - Configurable selection pressure

4. **Crossover**
   - Single-point crossover
   - Weight matrix recombination
   - Probability: 0.7 (default)

5. **Mutation**
   - Gaussian noise addition
   - Probability: 0.1 (default)
   - Mutation rate decay

## Weight Logger System

The weight logging system (`weights_logger.py`) provides comprehensive tracking and analysis of neural network weights during training.

### Features

1. **Weight Tracking**
   - Logs weights for each generation
   - Tracks fitness scores
   - Records training parameters

2. **Analysis Tools**
   - Weight distribution visualization
   - Fitness score trends
   - Performance correlation analysis

3. **Best Weight Identification**
   - Multiple evaluation metrics
   - Cross-validation across different seeds
   - Parameter sensitivity analysis

### Usage Example
```python
# Initialize logger
logger = WeightLogger("training_session_1")

# Log weights during training
logger.log_weights(generation, weights, fitness_score)

# Analyze results
best_weights = logger.get_best_weights(
    metric="average_score",
    num_seeds=5,
    game_params={"speed": 1.2, "spawn_rate": 0.8}
)
```

### Parameter Sensitivity
The weight logger helps identify how different parameters affect performance:
- Game speed
- Spawn rates
- Number of evaluation seeds
- Genetic algorithm parameters

## Usage

### Training Mode
```bash
# Basic training
python main.py -t

# Advanced training with parameters
python main.py -t -p 100 -g 100 --mutation-rate 0.1 --crossover-rate 0.7

# Training with weight logging
python main.py -t --log-weights training_session_1
```

### Play Mode
```bash
# Human play
python main.py

# AI play with specific weights
python main.py --weights best_weights.json

# Headless mode (no graphics)
python main.py --headless
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

## Contributing

Feel free to contribute to this project by:
1. Forking the repository
2. Creating a feature branch
3. Submitting a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
