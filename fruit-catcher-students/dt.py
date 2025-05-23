import numpy as np
import csv

class DecisionTree:
    """Decision Tree Classifier implementation.
    
    A binary decision tree that learns from training data to classify items as fruits (1) or bombs (-1).
    """

    def __init__(self, X, y, threshold=1.0, max_depth=None):
        """Initialize the Decision Tree.
        
        Args:
            X: List of lists containing the training features
            y: List of labels (1 for fruit, -1 for bomb)
            threshold: Float value for information gain threshold (default: 1.0)
            max_depth: Maximum depth of the tree (default: None)
        """
        self.x = X
        self.y = y
        self.threshold = threshold
        self.max_depth = max_depth
        self.tree = None
        self.feature_names = ['name', 'color', 'format']  # Store feature names

    def predict(self, x):
        """Predict if an item is a fruit or bomb.
        
        Args:
            x: List containing feature values ['name', 'color', 'format']
        
        Returns:
            int: 1 for fruit, -1 for bomb
        """
        if self.tree is None:
            raise ValueError("Tree not trained yet")
        
        return self._traverse_tree(self.tree, x)
    
    def _traverse_tree(self, tree, x):
        """Traverse the decision tree to make a prediction.
        
        This function works by recursively traversing a decision tree structure to classify an item:
        
        1. Base case: If we reach a leaf node (an integer value), return that classification
        2. For internal nodes (dictionaries):
           - Each node represents a feature test (e.g. 'color')
           - The branches represent possible values of that feature
           - We look up which branch to follow based on the input's feature value
           - Recursively traverse down that branch until reaching a leaf
        
        Example tree structure:
        {
            'color': {                    # Internal node testing 'color' feature
                'red': {                  # Branch for red items
                    'format': {           # Test format of red items
                        'circle': 1,      # Red circles are fruit (1)
                        'curved': -1      # Red curved items are bombs (-1) 
                    }
                },
                'blue': -1,              # Blue items are bombs
                'yellow': 1              # Yellow items are fruit
            }
        }
        
        Args:
            tree: Current node in decision tree (dict for internal nodes, int for leaves)
            x: List of feature values to classify
            
        Returns:
            int: 1 for fruit classification, -1 for bomb
        """
        # If we've reached a leaf node (integer classification)
        if not isinstance(tree, dict):
            return tree
        
        # Get the feature at this node
        feature = list(tree.keys())[0]
        # Get the index of this feature
        feature_idx = self.feature_names.index(feature)
        feature_value = x[feature_idx]
        
        # If we don't have this value in our tree, return the most common class
        if feature_value not in tree[feature]:
            # Return most common class (could be improved)
            return 1
            
        # Traverse down the appropriate branch
        return self._traverse_tree(tree[feature][feature_value], x)

def train_decision_tree(X, y, feature_names):
    """Train a decision tree classifier.
    
    Args:
        X: List of lists containing the training features
        y: List of labels (1 for fruit, -1 for bomb)
        feature_names: List of feature names
    
    Returns:
        DecisionTree: Trained decision tree classifier
    """
    dt = DecisionTree(X, y)
    dt.feature_names = feature_names  # Store feature names
    dt.tree = build_decision_tree(X, y, feature_names)
    print_tree(dt.tree)
    test_prediction()
    return dt

def find_best_feature(X, y, feature_names):
    """Find the best feature to split on based on information gain.
    
    Args:
        X: List of lists containing the training features
        y: List of labels (1 for fruit, -1 for bomb)
        feature_names: List of feature names
    
    Returns:
        str: Name of the feature with highest information gain
    """
    print("Training data:", X)
    print("Labels:", y)
    dataset_entropy = calculate_entropy(y)
    print("Dataset entropy:", dataset_entropy)

    # Get entropy for each feature (column)
    feature_entropies = []
    for feature_idx in range(len(X[0])):  # For each feature
        feature_column = [row[feature_idx] for row in X]  # Extract the feature column
        entropy = calculate_feature_entropy(feature_column, feature_idx, y)
        feature_entropies.append(entropy)
        print(f"Feature {feature_names[feature_idx]} entropy:", entropy)

    information_gains = calculate_information_gain(feature_entropies, dataset_entropy)
    print("Information gains by feature:", {name: gain for name, gain in zip(feature_names, information_gains)})

    best_feature_idx = np.argmax(information_gains)
    best_feature_name = feature_names[best_feature_idx]
    print(f"Best feature: {best_feature_name}")

    return best_feature_name

def calculate_entropy(y):
    """Calculate Shannon entropy of a label array.
    
    Args:
        y: List of labels (1 for fruit, -1 for bomb)
    
    Returns:
        float: Entropy value of the label distribution
    """
    unique, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    return -np.sum(probabilities * np.log2(probabilities))

def information_gain(y, y1, y2):
    """Calculate information gain for a binary split.
    
    Args:
        y: List of all labels before split
        y1: List of labels for first split
        y2: List of labels for second split
    
    Returns:
        float: Information gain value for the split
    """
    p = len(y1) / len(y)
    return calculate_entropy(y) - (p * calculate_entropy(y1) + (1 - p) * calculate_entropy(y2))

def calculate_feature_entropy(features, feature_idx, labels):
    """Calculate entropy for a specific feature.
    
    Args:
        features: List of values for a single feature
        feature_idx: Index of the feature
        labels: List of corresponding labels
    
    Returns:
        float: Entropy value for this feature
    """
    # Create a dictionary to store labels for each feature value
    feature_dict = {}
    
    # Group labels by feature values
    for i in range(len(features)):
        feature_val = features[i]  # Now this is a single value, not a list
        label = labels[i]
        
        if feature_val not in feature_dict:
            feature_dict[feature_val] = []
        feature_dict[feature_val].append(label)
    
    # Calculate weighted entropy
    total_samples = len(labels)
    weighted_entropy = 0
    
    for feature_val, feature_labels in feature_dict.items():
        prob = len(feature_labels) / total_samples
        weighted_entropy += prob * calculate_entropy(feature_labels)
    
    return weighted_entropy

def calculate_information_gain(feature_entropies, dataset_entropy):
    """Calculate information gain for all features.
    
    Args:
        feature_entropies: List of entropy values for each feature
        dataset_entropy: Float value of the entire dataset's entropy
    
    Returns:
        list: Information gain values for each feature
    """
    information_gains = []
    for i in range(len(feature_entropies)):
        information_gains.append(dataset_entropy - feature_entropies[i])
    return information_gains

def build_decision_tree(X, y, feature_names, depth=0, max_depth=5):
    """Build a decision tree recursively.
    
    Args:
        X: List of feature vectors
        y: List of labels
        feature_names: List of feature names
        depth: Current depth in the tree
        max_depth: Maximum depth to grow the tree
        
    Returns:
        dict: A decision tree represented as nested dictionaries
    """
    # Base cases
    if depth >= max_depth or len(set(y)) == 1:
        # Return most common label
        return max(set(y), key=y.count)
    
    # Calculate entropy of current dataset
    dataset_entropy = calculate_entropy(y)
    
    # Calculate entropy for each feature
    feature_entropies = []
    for i in range(len(X[0])):
        feature_values = [x[i] for x in X]
        feature_entropies.append(calculate_feature_entropy(feature_values, i, y))
    
    # Calculate information gain for all features
    gains = calculate_information_gain(feature_entropies, dataset_entropy)
    
    # Find feature with maximum information gain
    best_feature_idx = gains.index(max(gains))
    best_feature_name = feature_names[best_feature_idx]
    
    # Create tree node
    tree = {best_feature_name: {}}
    
    # Get unique values for best feature
    feature_values = set(x[best_feature_idx] for x in X)

    # Get unique values of the best feature to create branches
    
    # Create branches for each value
    for value in feature_values:
        # For each unique value of the best feature, create a branch in the tree
        # and recursively build subtrees for the remaining data that matches this value
        # Get indices where feature has this value

        indices = [i for i, x in enumerate(X) if x[best_feature_idx] == value]
        
        # Create subsets
        sub_X = [X[i] for i in indices]
        sub_y = [y[i] for i in indices]
        
        # Remove used feature from feature names
        remaining_features = feature_names.copy()
        remaining_features.pop(best_feature_idx)
        
        # Remove used feature from feature vectors
        sub_X = [[x[i] for i in range(len(x)) if i != best_feature_idx] for x in sub_X]
        
        # Recursively build subtree
        tree[best_feature_name][value] = build_decision_tree(sub_X, sub_y, remaining_features, depth + 1, max_depth)

        # Example of populated tree structure:
        # tree = {
        #     'color': {
        #         'red': {
        #             'format': {
        #                 'circle': 1,
        #                 'curved': -1
        #             }
        #         },
        #         'blue': -1,
        #         'yellow': 1
        #     }
        # }
    
    return tree

def print_tree(tree, indent=""):
    
    """
    Print a decision tree in a readable format.
    
    Args:
        tree: The decision tree to print
        indent: String for indentation level (used in recursion)
    """
    if isinstance(tree, dict):
        # Get the feature name (root of this subtree)
        feature = list(tree.keys())[0]
        print(f"{indent}{feature}")
        
        # Print all branches from this node
        for value, subtree in tree[feature].items():
            print(f"{indent}  ├─ {value} →")
            print_tree(subtree, indent + "  │  ")
    else:
        # Leaf node - print the classification
        print(f"{indent}  └─ Class: {tree}")

def test_prediction():
    """Test the decision tree predictions on items data."""
    # Load items data
    X = []
    y = []
    with open('items.csv') as f:
        reader = csv.reader(f, delimiter=';')
        headers = next(reader)  # Skip header row
        feature_names = headers[1:-1]  # Get feature names excluding id and is_fruit
        
        for row in reader:
            X.append([row[1], row[2], row[3]])  # name, color, format columns
            y.append(int(row[4]))  # is_fruit column
    
    # Load training data to build the tree
    train_X = []
    train_y = []
    with open('train.csv') as f:
        reader = csv.reader(f, delimiter=';')
        next(reader)  # Skip header
        for row in reader:
            train_X.append([row[1], row[2], row[3]])
            train_y.append(int(row[4]))
    
    # Create and train decision tree with training data
    dt = DecisionTree(train_X, train_y)
    dt.feature_names = feature_names
    dt.tree = build_decision_tree(train_X, train_y, feature_names)
    
    # Test predictions on items data
    print("\nPredictions for items:")
    for x, true_y in zip(X, y):
        prediction = dt.predict(x)
        print(f"Item: {x}, Predicted: {'Fruit' if prediction == 1 else 'Bomb'}, Actual: {'Fruit' if true_y == 1 else 'Bomb'}")