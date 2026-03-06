from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

from activation_functions import ActivationFunction


class Layer(ABC):
    def __init__(self, size: int) -> None:
        self._size: int = size

    @property
    def size(self) -> int:
        return self._size

    @abstractmethod
    def forward_pass(self, input_data: NDArray[np.float64]) -> NDArray[np.float64]:
        pass


class DenseLayer(Layer):
    def __init__(
        self, input_size: int, size: int, activation_function: ActivationFunction
    ) -> None:
        super().__init__(size)

        # TODO: Move weight initialization to separate functions to be passed in
        self._weights: NDArray[np.float64] = np.random.randn(input_size, size) * 0.01
        self._biases: NDArray[np.float64] = np.zeros((1, size))
        self._activation_function: ActivationFunction = activation_function

        # Cache for forward pass data
        self._input: NDArray[np.float64] = np.array([])
        self._pre_output: NDArray[np.float64] = np.array([])

    def forward_pass(self, input_data: NDArray[np.float64]) -> NDArray[np.float64]:
        self._input = input_data

        self._pre_output = np.dot(self._input, self._weights) + self._biases
        return self._activation_function.apply(self._pre_output)

    def backward_pass(
        self, output_gradient: NDArray[np.float64], learning_rate: float
    ) -> NDArray[np.float64]:
        chain_gradient: NDArray[np.float64] = (
            output_gradient * self._activation_function.derivative(self._pre_output)
        )

        input_gradient: NDArray[np.float64] = np.dot(chain_gradient, self._weights.T)
        weight_gradient: NDArray[np.float64] = np.dot(self._input.T, chain_gradient)
        bias_gradient: NDArray[np.float64] = np.sum(
            chain_gradient, axis=0, keepdims=True
        )

        self._weights -= weight_gradient * learning_rate
        self._biases -= bias_gradient * learning_rate

        return input_gradient
