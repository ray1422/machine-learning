"""
my_model.py
I implemented the polynomial regression model in this file by myself.
SK-learn base classes are imported for the compatibility for utils only without implementing the fitting algorithm.
"""

import typing
import numpy as np
from sklearn.base import RegressorMixin, MultiOutputMixin
from sklearn.linear_model._base import LinearModel


def my_transform(x: np.ndarray, degree: int) -> np.ndarray:
    """
    Transform the input data.
    :param x: The input data. 1d np.array with length n_data
    :return: The transformed data. np.array of shape (n_data, degree+1)
    """
    dim = degree + 1
    x2 = np.zeros((np.shape(x)[0], dim))
    for i in range(dim):
        x2[:, i] = np.reshape(x ** i, (-1))
    return x2


class MyLinearRegression(MultiOutputMixin, RegressorMixin, LinearModel):
    def __init__(self):
        super().__init__()
        self.coef_: typing.Optional[np.ndarray] = None

    def fit(self, x: np.ndarray, y: np.ndarray):
        """
        Fit the model to the data.
        :param x: The input data. np.array of shape (n_data, inp_dim)
        :param y: The excepted output data. np.array of shape (n_data, out_dim)
        """
        # self.coef_ = np.matmul(np.linalg.inv(np.matmul(x.T, x)), np.matmul(x.T, y))
        # self.coef_ = np.matmul(np.linalg.pinv(np.matmul(x.T, x)), np.matmul(x.T, y))
        # @ is matrix multiplication. I didn't know this syntax until GitHub Copilot told me.
        self.coef_ = np.linalg.pinv(x) @ y

    def predict(self, x: np.array) -> np.array:
        return np.matmul(x, self.coef_)


class MyRidgeRegression(MyLinearRegression):
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha

    def fit(self, x: np.ndarray, y: np.ndarray):
        """
        Fit the model to the data.
        :param x: The input data. np.array of shape (n_data, inp_dim)
        :param y: The excepted output data. np.array of shape (n_data, out_dim)
        """
        self.coef_ = np.matmul(np.linalg.pinv(x.T @ x + (self.alpha) / np.shape(x)[0] * np.eye(x.shape[1])), x.T @ y)

    def predict(self, x: np.array) -> np.array:
        return np.matmul(x, self.coef_)
