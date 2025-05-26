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

<<<<<<< Updated upstream
=======
## Neural Network Implementation (nn.py)

The neural network controller implements a feedforward neural network for game control decisions.

### Network Architecture

```
Input Layer (10)     Hidden Layer 1 (8)    Output Layer (1)
    [x]                 [x]                     [x]
    [x]                 [x]  
    [x]                 [x]  
    [x]                 [x]   
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

2. **Hidden Layer (8 neurons)**
   - ReLU activation
   - Learns complex patterns in item trajectories
   - Size calculated using rules of thumb:
     - Mean of input and output size
     - 2/3 of input size + output size
     - Square root of input size times output size

3. **Output Layer (1 neuron)**
   - Sigmoid activation
   - Outputs: -1 (move left) or 1 (move right)

## Genetic Algorithm Implementation (genetic.py)

The genetic algorithm optimizes the neural network weights through evolutionary processes.

### Algorithm Flow
```
Random Weight Initialization (-1 to 1)
      ↓
Multi-Seed Evaluation (12 seeds per individual)
      ↓
Tournament Selection (8 participants) + Elitism (5% preservation)
      ↓
Single-Point Crossover (70% probability)
      ↓
Gaussian Mutation (25% base rate, adaptive)
      ↓
Population Replacement
```

Each step in detail:
1. **Initial Population**
   - Random weight generation between -1 and 1
   - Population size configurable (default: 100)
   - Weights normalized to prevent extreme values

2. **Fitness Evaluation**
   - Each individual tested with 12 different random seeds
   - Average score calculated across all evaluations
   - Movement penalty applied for low variance
   - Minimum score capped at 0

3. **Selection**
   - Tournament Selection: 8 random individuals compete
   - Elitism: Top 5% of population preserved
   - Selection pressure balanced for exploration/exploitation

4. **Crossover**
   - Single-point crossover between parent pairs
   - 70% probability of crossover occurring
   - Weight matrices recombined at random point

5. **Mutation**
   - Base mutation rate: 25%
   - Gaussian noise addition to weights
   - Adaptive rate based on stagnation
   - Weights clamped to [-1, 1] range

6. **New Generation**
   - Combines elite individuals with offspring
   - Maintains population size
   - Preserves best solutions while exploring new ones

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

## Game Scoring System

The game uses a simple but effective scoring mechanism:

### Score Calculation
- +1 point for each fruit caught
- Game ends if:
  - A bomb is caught
  - Fruit limit is reached (default: 100 fruits)
  - All items are cleared after reaching fruit limit

### Fitness Evaluation
The genetic algorithm evaluates individuals using:
1. **Multiple Evaluations**
   - Each individual is tested with multiple random seeds
   - Default: 12 seeds per evaluation
   - Provides robust performance assessment

2. **Score Components**
   - Average score across all evaluations
   - Movement penalty (-2.0) if score variance < 1.0
   - Minimum score of 0 (no negative scores)

3. **Evaluation Parameters**
   - Fruit drop rate: Every 30 frames
   - Bomb drop rate: Every 100 frames
   - Maximum fruits: 100 (configurable)

### Weight Logger Analysis
The weight logger provides detailed performance metrics:
- Average score across multiple evaluations
- Minimum and maximum scores
- Score standard deviation
- Weight distribution statistics

## Usage

### Training Mode
```bash
# Basic training
python main.py -t

# Advanced training with parameters
python main.py -t -p 100 -g 100 -m 0.1 -c 0.7

# Training with weight logging
python main.py -t -l training_session_1
```

### Play Mode
```bash
# Human play
python main.py

# AI play with specific weights
python main.py -w best_individual.txt

# Headless mode (no graphics)
python main.py -l
```

>>>>>>> Stashed changes
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
