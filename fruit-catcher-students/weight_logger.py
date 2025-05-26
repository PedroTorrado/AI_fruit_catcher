import time
import os
from datetime import datetime
import numpy as np
import subprocess
import re
from main import load_ai_player, train_fruit_classifier

def evaluate_weights(num_evaluations=20):
    """Run game.py with current weights multiple times and get average score."""
    scores = []
    
    print(f"\nRunning {num_evaluations} evaluations...")
    for i in range(num_evaluations):
        try:
            # Run game.py with -l flag and current weights
            result = subprocess.run(['python', 'main.py', '-l', '-f', 'best_individual.txt'], 
                                 capture_output=True, 
                                 text=True)
            
            # Extract score from output
            match = re.search(r'Score: (\d+)', result.stdout)
            if match:
                score = int(match.group(1))
                scores.append(score)
                print(f"Evaluation {i+1}/{num_evaluations}: Score = {score}")
            else:
                print(f"Evaluation {i+1}/{num_evaluations}: Could not extract score")
                print("Game output:", result.stdout)  # Debug output
                
        except Exception as e:
            print(f"Error in evaluation {i+1}: {e}")
    
    if scores:
        avg_score = sum(scores) / len(scores)
        print(f"\nAverage score over {len(scores)} evaluations: {avg_score:.2f}")
        return avg_score, scores
    return None, []

def log_weights(weights, fitness_score=None):
    """Log weights and fitness score to a log file with timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Create logs directory if it doesn't exist
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    # Log file path
    log_file = 'logs/weight_history.log'
    
    # Calculate some statistics about the weights
    weights_array = np.array(weights)
    stats = {
        'mean': np.mean(weights_array),
        'std': np.std(weights_array),
        'min': np.min(weights_array),
        'max': np.max(weights_array),
        'positive_count': np.sum(weights_array > 0),
        'negative_count': np.sum(weights_array < 0)
    }
    
    # Run evaluations and get average score
    avg_score, all_scores = evaluate_weights()
    
    # Format the log entry
    log_entry = f"\n{'='*80}\n"
    log_entry += f"Timestamp: {timestamp}\n"
    if avg_score is not None:
        log_entry += f"Game Evaluation Results:\n"
        log_entry += f"  Average Score: {avg_score:.2f}\n"
        log_entry += f"  Min Score: {min(all_scores)}\n"
        log_entry += f"  Max Score: {max(all_scores)}\n"
        log_entry += f"  Score Std Dev: {np.std(all_scores):.2f}\n"
        log_entry += f"  All Scores: {all_scores}\n"
    if fitness_score is not None:
        log_entry += f"Fitness Score: {fitness_score:.2f}\n"
    log_entry += f"Weight Statistics:\n"
    log_entry += f"  Mean: {stats['mean']:.4f}\n"
    log_entry += f"  Std Dev: {stats['std']:.4f}\n"
    log_entry += f"  Min: {stats['min']:.4f}\n"
    log_entry += f"  Max: {stats['max']:.4f}\n"
    log_entry += f"  Positive Weights: {stats['positive_count']}\n"
    log_entry += f"  Negative Weights: {stats['negative_count']}\n"
    log_entry += f"Weights: {','.join(map(str, weights))}\n"
    
    # Append to log file
    with open(log_file, 'a') as f:
        f.write(log_entry)
    
    print(f"Logged weights to {log_file}")

def monitor_best_individual():
    """Monitor best_individual.txt for changes and log them."""
    last_modified = 0
    last_content = None
    
    print("Starting weight logger...")
    print("Monitoring best_individual.txt for changes...")
    
    while True:
        try:
            # Check if file exists
            if not os.path.exists('best_individual.txt'):
                time.sleep(1)
                continue
            
            # Get file modification time
            current_modified = os.path.getmtime('best_individual.txt')
            
            # If file has been modified
            if current_modified > last_modified:
                # Read the new content
                with open('best_individual.txt', 'r') as f:
                    content = f.read().strip()
                
                # If content has changed
                if content != last_content:
                    weights = [float(x) for x in content.split(',')]
                    log_weights(weights)
                    last_content = content
                
                last_modified = current_modified
            
            time.sleep(1)  # Check every second
            
        except KeyboardInterrupt:
            print("\nStopping weight logger...")
            break
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(1)

if __name__ == "__main__":
    monitor_best_individual() 