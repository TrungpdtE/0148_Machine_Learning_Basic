from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

# Load dataset
iris = load_iris()
X, y = iris.data, iris.target

# Create Decision Tree classifier using ID3 algorithm
clf = DecisionTreeClassifier(criterion='entropy')
clf = clf.fit(X, y)