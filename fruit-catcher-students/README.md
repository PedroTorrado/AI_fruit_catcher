# Fruit Catcher Game

This project was developed for the Artificial Intelligence class during my bachelor's course at ISCTE.

A Python-based game where players catch falling fruits while avoiding bombs, featuring both human playable mode and AI-controlled gameplay using neural networks and genetic algorithms.

## Description

Fruit Catcher is an interactive game where a basket moves horizontally at the bottom of the screen to catch falling fruits while avoiding bombs. The game includes:

- Human playable mode with keyboard controls
- AI player mode using neural networks
- Genetic algorithm for training the AI
- Decision tree-based fruit classification system
- Score tracking system

## Features

- Simple and intuitive gameplay
- Multiple game modes (Human/AI)
- Neural network-based AI player
- Genetic algorithm for AI training
- Real-time fruit classification
- Score tracking
- Graphical interface using Pygame

## Requirements

- Python 3.x
- Pygame
- NumPy
- Other dependencies (specified in requirements)

## Installation

1. Clone the repository
2. Install the required dependencies:
```bash
pip install pygame numpy
```

## Usage

### Running the Game

To start the game in normal mode:
```bash
python main.py
```

### Training the AI

To train the AI using genetic algorithm:
```bash
python main.py -t
```

Additional training parameters:
- `-p` or `--population`: Set population size (default: 100)
- `-g` or `--generations`: Set number of generations (default: 100)
- `-f` or `--file`: Specify file to store/load AI weights
- `-l` or `--headless`: Run without graphics

Example:
```bash
python main.py -t -p 200 -g 150 -f custom_weights.txt
```

## Game Controls

### Human Mode
- Left Arrow: Move basket left
- Right Arrow: Move basket right

### AI Mode
The AI automatically controls the basket based on trained neural network weights.

## Project Structure

- `main.py`: Main game entry point and AI training logic
- `game.py`: Core game mechanics and display
- `genetic.py`: Genetic algorithm implementation
- `nn.py`: Neural network architecture
- `dt.py`: Decision tree implementation for fruit classification
- `items.csv`: Item definitions
- `train.csv`: Training data for fruit classification
- `test.csv`: Test data for fruit classification
- `images/`: Game assets directory

## How it Works

1. **Game Mechanics**:
   - Fruits and bombs fall from the top of the screen
   - Player controls a basket at the bottom
   - Catch fruits to score points
   - Avoid bombs (game ends if caught)

2. **AI Implementation**:
   - Neural network processes game state
   - Genetic algorithm optimizes network weights
   - State includes basket position and item information

3. **Fruit Classification**:
   - Decision tree classifies items as fruits or bombs
   - Training data provided in CSV format
   - Real-time classification during gameplay

