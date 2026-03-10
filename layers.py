from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

from activation_functions import ActivationFunction, Linear


class Layer(ABC):
    def __init__(self, size: int) -> None:
        self._size: int = size

    @property
    def size(self) -> int:
        return self._size

    @abstractmethod
    def forward_pass(self, input_data: NDArray[np.float64]) -> NDArray[np.float64]:
        pass

    @abstractmethod
    def backward_pass(
        self, output_gradient: NDArray[np.float64], learning_rate: float
    ) -> NDArray[np.float64]:
        pass


class DenseLayer(Layer):
    def __init__(
        self, size: int, activation_function: ActivationFunction = Linear()
    ) -> None:
        super().__init__(size)

        self._weights: NDArray[np.float64] | None = None
        self._biases: NDArray[np.float64] = np.zeros((1, size))
        self._activation_function: ActivationFunction = activation_function

        self._input_cache: NDArray[np.float64] | None = None
        self._z_cache: NDArray[np.float64] | None = None  # Pre-activation output

    def forward_pass(self, input_data: NDArray[np.float64]) -> NDArray[np.float64]:
        if self._weights is None:
            # TODO: Allow passing in different initialization methods
            self._weights = np.random.randn(input_data.shape[-1], self.size) * 0.01

        self._input_cache = input_data

        self._z_cache = (self._input_cache @ self._weights) + self._biases
        return self._activation_function.apply(self._z_cache)

    def backward_pass(
        self, output_gradient: NDArray[np.float64], learning_rate: float
    ) -> NDArray[np.float64]:
        assert self._weights is not None
        assert self._input_cache is not None
        assert self._z_cache is not None

        z_gradient: NDArray[np.float64] = (
            output_gradient * self._activation_function.derivative(self._z_cache)
        )

        input_gradient: NDArray[np.float64] = z_gradient @ self._weights.T
        weight_gradient: NDArray[np.float64] = self._input_cache.T @ z_gradient
        bias_gradient: NDArray[np.float64] = np.sum(z_gradient, axis=0, keepdims=True)

        self._weights -= weight_gradient * learning_rate
        self._biases -= bias_gradient * learning_rate

        return input_gradient
