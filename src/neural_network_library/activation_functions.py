"""
Provides activation function classes to be used by neural network layers.
"""

from abc import ABC, abstractmethod
from typing import override

import numpy as np


class ActivationFunction(ABC):
    """
    Abstract base class for activation functions.
    """

    @abstractmethod
    def apply(self, values: np.ndarray) -> np.ndarray:
        """
        Applies activation function to a neuron's output data.

        Args:
            values (np.ndarray): Values to apply activation function to.

        Returns:
            np.ndarray: Output data in same shape as values with activation applied.
        """

    @abstractmethod
    def derivative(self, values: np.ndarray) -> np.ndarray:
        """
        Applies derivative of activation function.

        Args:
            values (np.ndarray): Values to apply derivative to.

        Returns:
            np.ndarray: Output gradient in same shape as values.
        """


class LeakyReLU(ActivationFunction):
    """
    Leaky Rectified Linear Unit activation function.

    Args:
        alpha (float): The slope for values <= 0, defaults to 0.01.
    """

    def __init__(self, alpha: float = 0.01):
        self._alpha: float = alpha

    @override
    def apply(self, values: np.ndarray) -> np.ndarray:
        return np.where(values > 0, values, values * self._alpha)

    @override
    def derivative(self, values: np.ndarray) -> np.ndarray:
        return np.where(values > 0, 1, self._alpha)


class Linear(ActivationFunction):
    """
    Linear activation function.

    Returns input data unchanged, typically for output layer.
    """

    @override
    def apply(self, values: np.ndarray) -> np.ndarray:
        return values

    @override
    def derivative(self, values: np.ndarray) -> np.ndarray:
        return np.ones_like(values)


class ReLU(ActivationFunction):
    """
    Rectified Linear Unit activation function.

    Returns 0 for values <= 0, and original values when > 0.
    """

    @override
    def apply(self, values: np.ndarray) -> np.ndarray:
        return np.where(values > 0, values, 0)

    @override
    def derivative(self, values: np.ndarray) -> np.ndarray:
        return np.where(values > 0, 1, 0)


class Sigmoid(ActivationFunction):
    """
    Sigmoid/Logistic activation function.

    Output values in range 0 to 1.
    """

    @override
    def apply(self, values: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-values))

    @override
    def derivative(self, values: np.ndarray) -> np.ndarray:
        sigmoid: np.ndarray = self.apply(values)
        return sigmoid * (1 - sigmoid)


class Tanh(ActivationFunction):
    """
    Hyperbolic Tangent activation function.

    Output values in range -1 to 1.
    """

    @override
    def apply(self, values: np.ndarray) -> np.ndarray:
        return np.tanh(values)

    @override
    def derivative(self, values: np.ndarray) -> np.ndarray:
        return 1 - np.tanh(values) ** 2
