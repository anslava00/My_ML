import numpy as np


class MyLinearRegression:
    def __init__(self, fit_intercept=True):
        self.w = []
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        y_train = np.array(y)
        X_train = np.array(X)
        if self.fit_intercept:
            X_train = np.hstack((X_train, np.ones((X_train.shape[0], 1))))

        self.w = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train
        return self

    def predict(self, X):
        X_test = np.array(X)
        if self.fit_intercept:
            X_test = np.hstack((X, np.ones((X_test.shape[0], 1))))

        y_pred = X_test @ self.w

        return y_pred

    def get_weights(self):
        return self.w
