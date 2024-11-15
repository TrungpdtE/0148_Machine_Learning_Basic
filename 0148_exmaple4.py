from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

# Load dataset
iris = load_iris()
X, y = iris.data, iris.target

# Create Decision Tree classifier with pruning
clf = DecisionTreeClassifier(max_depth=3, min_samples_split=4, min_samples_leaf=2)
clf = clf.fit(X, y)