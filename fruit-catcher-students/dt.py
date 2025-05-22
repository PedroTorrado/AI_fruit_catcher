import numpy as np
import pandas as pd

class DecisionTree:

    def __init__(self, X, y, threshold=1.0, max_depth=None): # Additional optional arguments can be added, but the default value needs to be provided
        # Implement this
        pass

    def predict(self, x): # (e.g. x = ['apple', 'green', 'circle'] -> 1 or -1)
        # Implement this
        pass

def format_data(filename, features):    
    df = pd.read_csv(filename, delimiter=';')
    # First save the is_fruit column
    is_fruit = df['is_fruit']
    
    # Format the feature columns
    for column in features:
        unique_values = df[column].unique()
        mapping = {value: idx + 1 for idx, value in enumerate(sorted(unique_values))}
        df[column] = df[column].map(mapping)
    
    # Restore the is_fruit column
    df['is_fruit'] = is_fruit
    return df

def calculate_entropy(data, parameter, objective_value):

    if objective_value != None:
        data = data[data['is_fruit'] == objective_value]

    unique_values = sorted(data[parameter].unique())  # Get sorted unique values
    count = [0] * len(unique_values)
    # ? print("Unique values:", [int(x) for x in unique_values])  # Convert np.int64 to regular int for cleaner output
    
    # Create a mapping from value to index
    value_to_index = {val: idx for idx, val in enumerate(unique_values)}
    
    for value in data[parameter]:
        count[value_to_index[value]] += 1
    
    # Calculate probabilities
    probability = []
    for i in range(len(count)):
        prob = count[i] * 1.0 / len(data)
        probability.append(prob)
    
    # ? print("Counts:", count)
    # ? print("Probabilities:", probability)
    
    # Calculate entropy
    entropy = 0.0
    for prob in probability:
        if prob > 0:  # Avoid log(0)
            entropy -= prob * np.log2(prob)
    
    return entropy

def calculate_information_gain(data, features, dataset_entropy):

    objective_values = data['is_fruit'].unique()
    
    entropies = {}
    for value in objective_values:
        for feature in features:
            entropies[feature] = dataset_entropy - calculate_entropy(data, feature, value)
    
    return entropies

def select_best_feature(information_gains):
    best_feature = max(information_gains, key=information_gains.get)
    return best_feature

def train_decision_tree(features, objective, filename):
    print("Training decision tree...")
    formated_data = format_data(filename, features)

    dataset_entropy = calculate_entropy(formated_data, objective[0], None)

    information_gains = calculate_information_gain(formated_data, features, dataset_entropy)
    best_feature = select_best_feature(information_gains)
    print(best_feature)

    return DecisionTree(features, objective)

if __name__ == '__main__':
    print("Testing decision tree training...")
    # Load data from train.csv
    features = ['name', 'color', 'format']
    objective = ['is_fruit']
    train_decision_tree(features, objective, 'train.csv')