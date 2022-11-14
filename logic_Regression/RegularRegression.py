from MyLogicRegression import MyLogicRegression
import numpy as np


class MyElasticLogisticRegression(MyLogicRegression):
    def __init__(self, l1_coef=.2, l2_coef=.2):
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef
        self.w = None

    def get_grad(self, X_batch, y_batch, predictions):
        grad_basic = X_batch.T @ (predictions - y_batch)

        grad_l1 = self.l1_coef * np.sign(self.w)
        grad_l1[0] = 0
        grad_l2 = 2 * self.l2_coef * self.w
        grad_l2[0] = 0

        assert grad_l1[0] == grad_l2[0] == 0, "Bias в регуляризационные слагаемые не входит!"
        assert grad_basic.shape == grad_l1.shape == grad_l2.shape == (
            X_batch.shape[1],), "Градиенты должны быть столбцом из k_features + 1 элементов"

        return grad_basic + grad_l1 + grad_l2
