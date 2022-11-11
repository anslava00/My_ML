# Функция для расчета среднего значения
from numpy import matrix, dot, row_stack, column_stack


def mean_value(array: list):
    sum = 0
    for val in array:
        sum += val
    return sum / len(array)


# Функция для нахождения моды
def moda_value(array: list):
    moda = {}
    for val in array:
        try:
            moda[val] += 1
        except:
            moda[val] = 1
    max_moda = {'i': [], 'max': 0}
    for key, val in moda.items():
        if val > max_moda['max']:
            max_moda['i'] = [key]
            max_moda['max'] = val
        elif val == max_moda['max']:
            max_moda['i'].append(key)
    return max_moda['i']


# Функция расчета дисперсии(0-ГС, 1-Выборка)
def disp(array: list, type=0):
    sum = 0
    mean = mean_value(array)
    for val in array:
        sum += ((val - mean) ** 2) / (len(array) - type)
    return sum


# Функция расчета среднеквадратического или среднего отклонения(0-ГС, 1-Выборка)
def mean_qvad(array, type=0):
    print(array)
    return disp(array, type) ** .5


# Функция для расчета размаха
def scope_range(array):
    return max(array) - min(array)


# Функция для расчета z-index
def z_scale(array):
    z = []
    mean = mean_value(array)
    q_mean = mean_qvad(array)
    for val in array:
        z.append((val - mean) / q_mean)
    return z


# Функция для перемножения матриц
def matrix_mult(X, Y):
    Z = []
    for i in range(len(X)):
        Z.append([])
        for j in range(len(Y[0])):
            cell = 0
            for k in range(len(X[0])):
                cell += X[i][k] * Y[k][j]
            Z[-1].append(cell)
    return Z


# Функция для транспонирования матрицы
def matrix_transpose(X):
    X_t = []
    for j in range(len(X[0])):
        X_t.append([])
        for i in range(len(X)):
            X_t[-1].append(X[i][j])
    return X_t


# Функция для нахождения обратной матрицы
def matrix_revers(X):
    A = matrix(X)
    Ar1 = 1 / A[:1, :1]
    for k in range(1, len(X)):
        V = A[k, : k]
        U = A[: k, k]
        ak = (A[k, k] - dot(dot(V, Ar1), U))[0, 0]
        rk = dot(-1 / ak, dot(Ar1, U))
        qk = dot(-1 / ak, dot(V, Ar1))
        B = Ar1 - dot(dot(Ar1, U), qk)
        Ar1 = (row_stack(
            (column_stack((B, rk)), column_stack((qk, 1 / ak)))
        ))
    return Ar1.tolist()
