import numpy as np
from collections import Counter

def entropy(y):
    hist = np.bincount(y)
    ps = hist / len(y)
    return -np.sum([p * np.log2(p) for p in ps if p > 0])

def information_gain(X, y, feature_index):
    parent_entropy = entropy(y)
    values, counts = np.unique(X[:, feature_index], return_counts=True)
    weighted_entropy = np.sum([(counts[i] / np.sum(counts)) * entropy(y[X[:, feature_index] == v]) for i, v in enumerate(values)])
    return parent_entropy - weighted_entropy

# Example usage
X = np.array([[0, 0], [1, 1], [1, 0], [0, 1]])
y = np.array([0, 1, 1, 0])
print("Entropy of y:", entropy(y))
print("Information Gain for feature 0:", information_gain(X, y, 0))