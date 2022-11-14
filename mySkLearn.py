import numpy as np

from stepic.myModel.test import myNumPy as mnp


# Функция расчета ковариации
def covariation(array1, array2, V_type=0):
    if len(array1) != len(array2):
        return 0
    mean1 = mnp.mean_value(array1)
    mean2 = mnp.mean_value(array2)
    sum = 0
    for i in range(len(array1)):
        sum += (array1[i] - mean1) * (array2[i] - mean2)
    return sum / (len(array1) - V_type)


# Функция расчета кореляции
def corelation(array1, array2, V_type=0):
    std1 = mnp.mean_qvad(array1, type=V_type)
    std2 = mnp.mean_qvad(array2, type=V_type)
    cov = covariation(array1, array2, V_type=V_type)
    return cov / (std2 * std1)


# Функция для расчета коэффициентов b0 b1
def lin_reg_one(X, Y, V_type=0):
    stdX = mnp.mean_qvad(X, type=V_type)
    stdY = mnp.mean_qvad(Y, type=V_type)
    meanX = mnp.mean_value(X)
    meanY = mnp.mean_value(Y)
    cor = corelation(X, Y, V_type=V_type)
    b1 = stdY * cor / stdX
    b0 = meanY - b1 * meanX
    return b0, b1


def mse(y_predict, y_true):
    y_p = np.array(y_predict)
    y_t = np.array(y_true)
    sum_mse = 0
    for i in range(y_p.shape[0]):
        sum_mse += (y_t[i] - y_p[i]) ** 2
    return sum_mse / y_p.shape[0]

def generate_batches(X, y, batch_size):
    """
    param X: np.array[n_objects, n_features] --- матрица объекты-признаки
    param y: np.array[n_objects] --- вектор целевых переменных
    """
    assert len(X) == len(y)
    np.random.seed(42)
    X = np.array(X)
    y = np.array(y)
    perm = np.random.permutation(len(X))

    for batch_start in range(0,len(X), batch_size):
        if batch_start + batch_size <= len(X) - 1:
            yield X[perm[batch_start:batch_start + batch_size]], y[perm[batch_start:batch_start + batch_size]]

# Класс работает с DataFrame, не с list
class Liner_regres:
    def __init__(self, data, dependent_var, type_b='matrix'):
        self.columns_X = list(data.loc[:, data.columns != dependent_var].columns)
        self.columns_Y = data.loc[:, dependent_var].name
        self.Y = [list(data.loc[:, dependent_var])]
        self.X = [list(data[col]) for col in data.loc[:, data.columns != dependent_var]]
        self.X.insert(0, [1 for i in range(len(self.X[0]))])
        if type_b == 'matrix':
            self.b = self.b_coff_matrix()

    def b_coff_matrix(self):
        X = mnp.matrix_transpose(self.X)
        Y = mnp.matrix_transpose(self.Y)
        XTX = mnp.matrix_mult(mnp.matrix_transpose(X), X)
        XTY = mnp.matrix_mult(mnp.matrix_transpose(X), Y)
        RXTX = mnp.matrix_revers(XTX)
        return mnp.matrix_mult(RXTX, XTY)

    def predict(self, data):
        pred = self.b[0][0]
        print(data)
        for i in range(1):
            pred += data[i] * self.b[i + 1][0]
        return pred

    def predictions(self):
        pred = []
        for i in range(len(self.X[0])):
            pred.append(self.predict([self.X[j][i] for j in range(1, len(self.X))]))
        return pred

    def std_err(self):
        pred = self.predictions()
        true_val = self.Y[0]
        sum_val = 0
        for i in range(len(pred)):
            sum_val += (pred[i] - true_val[i]) ** 2
        return sum_val / (len(pred) - 2)

    def print_coff_b(self):
        print(self.std_err())
        # print(self.X)
        # print(self.b)
        # print(self.predict([55.4, 71.3, 79.9, 14.2]))
        # print(self.Y)

        # print(pd.DataFrame(self.b,
        #                    index=['Intersept', *self.columns_X],
        #                    columns=['Estimate'])
        #       )
