from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray


class ActivationFunction(ABC):
    @abstractmethod
    def apply(self, values: NDArray[np.float64]) -> NDArray[np.float64]:
        pass

    @abstractmethod
    def derivative(self, values: NDArray[np.float64]) -> NDArray[np.float64]:
        pass


class LeakyReLU(ActivationFunction):
    def __init__(self, alpha: float = 0.01):
        self._alpha: float = alpha

    def apply(self, values: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.where(values > 0, values, values * self._alpha)

    def derivative(self, values: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.where(values > 0, 1, self._alpha)


class Linear(ActivationFunction):
    def apply(self, values: NDArray[np.float64]) -> NDArray[np.float64]:
        return values

    def derivative(self, values: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.ones_like(values)


class ReLU(ActivationFunction):
    def apply(self, values: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.where(values > 0, values, 0)

    def derivative(self, values: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.where(values > 0, 1, 0)


class Sigmoid(ActivationFunction):
    def apply(self, values: NDArray[np.float64]) -> NDArray[np.float64]:
        return 1 / (1 + np.exp(-values))

    def derivative(self, values: NDArray[np.float64]) -> NDArray[np.float64]:
        sigmoid: NDArray[np.float64] = self.apply(values)
        return sigmoid * (1 - sigmoid)


class Tanh(ActivationFunction):
    def apply(self, values: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.tanh(values)

    def derivative(self, values: NDArray[np.float64]) -> NDArray[np.float64]:
        return 1 - np.tanh(values) ** 2
