import matplotlib.pyplot as plt

class SimpleLinearRegression:

    def __init__(self):
        self.slope = None
        self.intercept = None

    def fit(self , x , y):
        n = len(x)
        mean_x = sum(x) / n
        mean_y = sum(y) / n

        numerator = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
        denominator = sum((x[i] - mean_x) ** 2 for i in range(n))

        self.slope = numerator / denominator

        self.intercept = mean_y - self.slope * mean_x

    def predict(self, X):
        if self.slope is None or self.intercept is None:
            raise Exception("model not trained yet")

        return [self.slope * x + self.intercept for x in X ] 

    def r_squared(self,X, y):
        if self.slope is None or self.intercept is None:
            raise Exception("model not trained yet")
        
        y_mean = sum(y) / len(y)
        y_pred = self.predict(X)

        ss_res = sum((y[i] - y_pred[i]) ** 2 for i in range(len(y)))
        ss_tot = sum((y[i] - y_mean) ** 2 for i in range(len(y)))

        r_squared = 1 - (ss_res/ss_tot)

        return r_squared




X_train = [1, 2]
y_train = [8.3 , 8.1]

model = SimpleLinearRegression()
model.fit(X_train , y_train)

x_test = [3,4,5,6,7,8]
y_pred = model.predict(x_test)

print("Slope:" , model.slope)
print("Intercept:" , model.intercept)


r2_score = model.r_squared(X_train , y_train)
print("R_squared" , r2_score)


plt.scatter(X_train , y_train , label = "Training Data")

plt.plot(x_test,y_pred , color = 'red' , label = 'Regression Line')
plt.xlabel('X')
plt.ylabel('Y')
plt.title("Simple Linear regression")
plt.legend()
plt.grid(True)
plt.savefig("Test")
plt.show()