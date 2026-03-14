"""
Provides layer classes for a neural network.
"""

from abc import ABC, abstractmethod
from typing import override

import numpy as np

from .activation_functions import ActivationFunction, Linear


class Layer(ABC):
    """
    Abstract base class for layers.
    """

    @abstractmethod
    def forward(self, input_data: np.ndarray, *, is_training: bool) -> np.ndarray:
        """
        Calculates the output values for this layer.

        Args:
            input_data (np.ndarray): Input received from previous layer.
            is_training (bool): Whether this pass is part of training

        Returns:
            np.ndarray: Layer output.
        """

    @abstractmethod
    def backward(
        self, output_gradient: np.ndarray, *, learning_rate: float
    ) -> np.ndarray:
        """
        Calculates input gradient and updates weights/biases.

        Args:
            output_gradient (np.ndarray): Gradient of loss with respect to this layer's output.
            learning_rate (float): How much to adjust weights/biases in response to gradient.
        """


class DenseLayer(Layer):
    """
    A fully connected layer.

    Args:
        size (int): The number of neurons to use in this layer.
        activation_function (ActivationFunction): Activation function to apply to output,
            defaults to Linear
    """

    def __init__(
        self, size: int, activation_function: ActivationFunction = Linear()
    ) -> None:
        self._size: int = size

        self._weights: np.ndarray | None = None
        self._biases: np.ndarray = np.zeros((1, size))
        self._activation_function: ActivationFunction = activation_function

        self._input_cache: np.ndarray | None = None
        self._z_cache: np.ndarray | None = None  # Pre-activation output

    @override
    def forward(self, input_data: np.ndarray, *, is_training: bool) -> np.ndarray:
        if self._weights is None:
            # TODO: Allow passing in different initialization methods
            self._weights = np.random.randn(input_data.shape[-1], self._size) * 0.01

        self._input_cache = input_data

        self._z_cache = (self._input_cache @ self._weights) + self._biases
        return self._activation_function.apply(self._z_cache)

    @override
    def backward(
        self, output_gradient: np.ndarray, *, learning_rate: float
    ) -> np.ndarray:
        assert self._weights is not None
        assert self._input_cache is not None
        assert self._z_cache is not None

        z_gradient: np.ndarray = output_gradient * self._activation_function.derivative(
            self._z_cache
        )

        input_gradient: np.ndarray = z_gradient @ self._weights.T
        weight_gradient: np.ndarray = self._input_cache.T @ z_gradient
        bias_gradient: np.ndarray = np.sum(z_gradient, axis=0, keepdims=True)

        self._weights -= weight_gradient * learning_rate
        self._biases -= bias_gradient * learning_rate

        return input_gradient


class DropoutLayer(Layer):
    """
    Randomly masks out neurons during training.

    Args:
        dropout_rate (float): Probability of a neuron being zeroed out (default 0.5).
    """

    def __init__(self, dropout_rate: float = 0.5) -> None:
        self._dropout_rate: float = dropout_rate
        self._mask: np.ndarray | None = None

    @override
    def forward(self, input_data: np.ndarray, *, is_training: bool) -> np.ndarray:
        if not is_training:
            return input_data

        p: float = 1 - self._dropout_rate
        self._mask = np.random.binomial(1, p, size=input_data.shape) / p
        return input_data * self._mask

    @override
    def backward(
        self, output_gradient: np.ndarray, *, learning_rate: float
    ) -> np.ndarray:
        return output_gradient * self._mask
