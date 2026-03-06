from abc import ABC, abstractmethod
from typing import Self

import numpy as np
import numpy.typing as npt

from activation_functions import ActivationFunction

class Layer(ABC):
    def __init__(self, size: int) -> None:
        self._size: int = size

    @property
    def size(self) -> int:
        return self._size

    @abstractmethod
    def forward_pass(self, input_data: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        pass

class DenseLayer(Layer):
    def __init__(
            self, input_size: int, size: int,
            activation_function: ActivationFunction) -> None:
        super().__init__(size)

        # TODO: Move weight initialization to separate functions to be passed in
        self._weights: npt.NDArray[np.float64] = np.random.randn(input_size, size) * 0.01
        self._biases: npt.NDArray[np.float64] = np.zeros((1, size))
        self._activation_function: ActivationFunction = activation_function

        # Cache for forward pass data
        self._input: npt.NDArray[np.float64] | None = None
        self._output: npt.NDArray[np.float64] | None = None

    def forward_pass(self, input_data: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        self._input = input_data

        pre_output = np.dot(self._input, self._weights) + self._biases
        self._output = self._activation_function.apply(pre_output)
        return self._output
