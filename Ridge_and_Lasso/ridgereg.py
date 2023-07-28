import numpy as np



def ridge_regression(X, y, alpha):
    p = X.shape
    XTX = np.dot(X.T, X)
    XTX += alpha * np.identity(p)
    XTX_inv = np.linalg.inv(XTX)
    XTX_inv_XT = np.dot(XTX_inv, X.T)
    beta_hat = np.dot(XTX_inv_XT, y)
    return beta_hat

X = np.array([[1,2] , [1,3] , [1,5]] , dtype=np.float64)
y = np.array([[4] , [5] , [7]] , dtype=np.float64)
lamb = 0.1

beta = ridge_regression(X , y , lamb)
print(beta)