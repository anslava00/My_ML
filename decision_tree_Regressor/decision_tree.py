from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


def predict(y):
    return sum(y) / len(y)


def mse(y_true, y_pred):
    sum_y = 0
    for i in range(len(y_true)):
        sum_y += (y_true[i] - y_pred) ** 2
    return sum_y / len(y_true)


class DecisionTreeRegressor:
    def __init__(self, max_depth=10, min_samples_leaf=1, level_tree=0):
        self.value = None
        self.sample = None
        self.mse_value = None
        self.IG = None
        self.left = None
        self.right = None
        self.level_tree = level_tree
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf

    def fit(self, X, y):
        self.sample = len(X)
        y = np.array(y)
        self.value = predict(y)
        self.mse_value = mse(y, self.value)
        if len(y) == self.min_samples_leaf or self.level_tree == self.max_depth:
            return 0
        train_X_y = np.column_stack((np.array(X), np.array(y)))
        max_IG = None
        for j in range(train_X_y.shape[1] - 1):
            unique_val = np.unique(train_X_y[:, j])
            for i in range(len(unique_val) - 1):
                left_branch = train_X_y[train_X_y[:, j] <= unique_val[i], -1]
                right_branch = train_X_y[train_X_y[:, j] > unique_val[i], -1]
                if len(left_branch) < self.min_samples_leaf or len(right_branch) < self.min_samples_leaf:
                    continue
                IG = self.mse_value - (len(left_branch) / self.sample * mse(left_branch, predict(left_branch))
                                       + len(right_branch) / self.sample * mse(right_branch, predict(right_branch)))
                if not max_IG or max_IG['IG'] < IG:
                    max_IG = {'index': j, 'value': unique_val[i], 'IG': IG}
        if not max_IG:
            return 0
        self.IG = max_IG

        left_branch = train_X_y[train_X_y[:, self.IG['index']] <= self.IG['value'], :]
        right_branch = train_X_y[train_X_y[:, self.IG['index']] > self.IG['value'], :]
        self.left = DecisionTreeRegressor(min_samples_leaf=self.min_samples_leaf, max_depth=self.max_depth, level_tree=self.level_tree + 1)
        self.right = DecisionTreeRegressor(min_samples_leaf=self.min_samples_leaf, max_depth=self.max_depth, level_tree=self.level_tree + 1)

        self.left.fit(left_branch[:, :-1], left_branch[:, -1])
        self.right.fit(right_branch[:, :-1], right_branch[:, -1])

    def predict(self, X):
        print('some')

    def print_tree(self):
        if self.left or self.right:
            print('deep_tree:', self.level_tree, '| node:', self.IG)
        else:
            print('deep_tree:', self.level_tree, '| leaf:', self.value, '| sample:', self.sample)

        if self.left:
            self.left.print_tree()
        if self.right:
            self.right.print_tree()


def main():
    df_house = pd.DataFrame(fetch_california_housing(as_frame=True).frame)
    train_df_house = df_house.loc[:, ['HouseAge', 'Population', 'MedHouseVal']].head(50)
    # tree_model = DecisionTreeRegressor(min_samples_leaf=2, max_depth=3)
    # tree_model.fit(X=train_df_house.loc[:, ['HouseAge', 'Population']], y=train_df_house['MedHouseVal'])
    # tree_model.print_tree()

    house_train, house_test, house_val_train, house_val_test = train_test_split(
        train_df_house.loc[:, train_df_house.columns != 'MedHouseVal'],
        train_df_house.loc[:, ['MedHouseVal']],
        test_size=0.33, random_state=42)
    tree_model = DecisionTreeRegressor(min_samples_leaf=3, max_depth=5)
    tree_model.fit(X=house_train, y=house_val_train)
    tree_model.print_tree()
    print('-----------------------')
    tree_model.predict(house_test)


if __name__ == '__main__':
    main()
