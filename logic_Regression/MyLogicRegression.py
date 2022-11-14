import numpy as np


def generate_batches(X, y, batch_size):
    assert len(X) == len(y)
    np.random.seed(42)
    X = np.array(X)
    y = np.array(y)
    perm = np.random.permutation(len(X))

    for batch_start in range(0, len(X), batch_size):
        if batch_start + batch_size <= len(X) - 1:
            yield X[perm[batch_start:batch_start + batch_size]], y[perm[batch_start:batch_start + batch_size]]


class MyLogicRegression(object):
    def __init__(self):
        self.w = None

    def fit(self, X, y, epochs=10, lr=0.1, batch_size=100):
        if self.w is None:
            np.random.seed(42)
            self.w = np.random.randn(X.shape[1] + 1)

        X_train = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)

        losses = []

        for i in range(epochs):
            for X_batch, y_batch in generate_batches(X_train, y, batch_size):
                predictions = self._predict_proba_internal(X_batch)
                loss = self.__loss(y_batch, predictions)
                losses.append(loss)

                self.w -= lr * self.grad(X_batch, y_batch, predictions)

        return losses

    def grad(self, X_batch, y_batch, predictions):
        grad = X_batch.T @ (predictions - y_batch) / X_batch.shape[0]
        return grad

    def predict_proba(self, X):
        X_train = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
        return self.sigmoid(self.logit(X_train, self.w))

    def _predict_proba_internal(self, X):
        return self.sigmoid(self.logit(X, self.w))

    def predict(self, X, threshold=0.5):
        return self.predict_proba(X) >= threshold

    def logit(self, X, w):
        return np.dot(X, w)

    def sigmoid(self, X):
        return 1 / (1 + np.exp(-X))

    def __loss(self, y, p):
        p = np.clip(p, 1e-10, 1 - 1e-10)
        return -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
