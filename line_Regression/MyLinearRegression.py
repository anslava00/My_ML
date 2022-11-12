import numpy as np


class MyLinearRegression:
    def __init__(self, fit_intercept=True):
        self.w = []
        self.fit_intercept = fit_intercept

    # type : matrix || grad
    def fit(self, X, y, type_fit='matrix'):
        y_train = np.array(y)
        X_train = np.array(X)
        if self.fit_intercept:
            X_train = np.hstack((X_train, np.ones((X_train.shape[0], 1))))
        if type_fit == 'matrix':
            self.fit_liner(X_train, y_train)
        elif type_fit == 'grad':
            self.fit_grad(X_train, y_train)
        return self

    def fit_liner(self, X_train, y_train):
        self.w = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train

    def fit_grad(self, X_train, y_train, epsilon=0.001, lr=0.01):
        self.w = np.zeros((X_train.shape[1], 1))
        loss = [self.mse_func(self.predict(X_train[:, :-1]), y_train)[0]]
        while len(loss) < 2 or abs(loss[-2] - loss[-1]) > epsilon:
            self.w -= lr * self.grad_func(X_train, y_train)
            loss.append(self.mse_func(self.predict(X_train[:, :-1]), y_train)[0])
        print('Количество итераций:', len(loss))

    def grad_func(self, X_train, y_train):
        grad = 2 * (self.predict(X_train[:, :-1]) - y_train[:, np.newaxis]).T @ X_train
        return grad.T / X_train.shape[0]

    def predict(self, X):
        X_test = np.array(X)
        if self.fit_intercept:
            X_test = np.hstack((X, np.ones((X_test.shape[0], 1))))
        y_pred = X_test @ self.w

        return y_pred

    def get_weights(self):
        return self.w

    def mse_func(self, y_pred, y):
        sum_err = 0
        for i in range(len(y)):
            sum_err += (y_pred[i] - y[i]) ** 2
        return sum_err / len(y)
