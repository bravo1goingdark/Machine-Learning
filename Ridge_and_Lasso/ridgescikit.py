import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge

# Example data (same as the dry run)
X = np.array([[1, 2], [1, 3], [1, 5]], dtype=np.float64)
y = np.array([[4], [5], [7]], dtype=np.float64)

# Create a Ridge regression model with alpha (regularization parameter) set to 0.1
ridge_model = Ridge(alpha=0.1)

# Fit the model to the data
ridge_model.fit(X, y)

# Get the ridge regression coefficients
beta_hat = ridge_model.coef_[0]
beta_0 = ridge_model.intercept_[0]

# Generate points to plot the regression line
x_points = np.linspace(1, 6, 100)
y_predicted = beta_0 + beta_hat[1] * x_points

# Plot the original data points
plt.scatter(X[:, 1], y, color='red', label='Data Points')

# Plot the regression line obtained from ridge regression
plt.plot(x_points, y_predicted, color='blue', label='Ridge Regression Line')

plt.xlabel('Feature')
plt.ylabel('Target')
plt.title('Ridge Regression using scikit-learn')
plt.legend()
plt.grid(True)
plt.savefig("Test")
plt.show()
