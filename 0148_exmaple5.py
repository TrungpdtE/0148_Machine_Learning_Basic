from sklearn.tree import DecisionTreeRegressor
import numpy as np
import matplotlib.pyplot as plt

# Generate some data
X = np.arange(10).reshape(-1, 1)
y = np.sin(X).ravel()

# Create Decision Tree regressor
regressor = DecisionTreeRegressor()
regressor = regressor.fit(X, y)

# Predict
X_test = np.arange(0, 10, 0.1).reshape(-1, 1)
y_pred = regressor.predict(X_test)

# Plot
plt.figure()
plt.scatter(X, y, s=20, edgecolor="black", c="darkorange", label="data")
plt.plot(X_test, y_pred, color="cornflowerblue", label="prediction")
plt.xlabel("data")
plt.ylabel("target")
plt.title("Decision Tree Regression")
plt.legend()
plt.show()