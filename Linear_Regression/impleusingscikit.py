import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


X_train = np.array([1,2]).reshape(-1,1)
Y_train = np.array([8.3 , 8.1])


model = LinearRegression()
model.fit(X_train , Y_train)


x_test = np.array([3,4,5,6,7,8]).reshape(-1,1)
y_pred = model.predict(x_test)

print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)

# Calculate R2 score
r2_score = model.score(X_train, Y_train)
print("R2 score:", r2_score)

# Plot the data points and the regression line
plt.scatter(X_train, Y_train, label='Training Data')
plt.plot(x_test, y_pred, color='red', label='Regression Line')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.title('Simple Linear Regression using Scikit-learn')
plt.grid(True)
plt.savefig("scikit")
plt.show()