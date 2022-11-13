import numpy as np


class MyLogicRegression:
    def __init__(self):
        self.w = []

    def logit(self, X, w):
        return X @ w

    def sigmoid(self, X):
        return 1 / (1 + np.exp(-X))

    # type : matrix || grad
    def fit(self, X, y, epsilon=0.001, lr=0.01):
        y_train = np.array(y)
        X_train = np.array(X)
        X_train = np.concatenate((np.ones((X_train.shape[0], 1)), X), axis=1)

        self.w = np.zeros((X_train.shape[1], 1))

        for i in range(500):
            print(self.loss(y_train, self.sigmoid(self.logit(X_train, self.w))))
            self.w -= lr * self.grad_func(X_train, y_train)

        return self

    def grad_func(self, X_train, y_train):
        grad = X_train.T @ (self.sigmoid(self.logit(X_train, self.w)) - y_train[:, np.newaxis])
        return grad / X_train.shape[0]

    def predict_proba(self, X):
        X_new = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
        return self.sigmoid(self.logit(X_new, self.w))

    def predict(self, X, threshold=0.5):
        return self.predict_proba(X) >= threshold

    def get_weights(self):
        return self.w

    def loss(self, y, p):
        p = np.clip(p, 1e-10, 1 - 1e-10)
        return np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
