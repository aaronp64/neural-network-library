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

class InputLayer(Layer):
    def __init__(self, size: int) -> None:
        super().__init__(size)

class DenseLayer(Layer):
    def __init__(
            self, prev_layer: Layer, size: int,
            activation_function: ActivationFunction) -> None:
        super().__init__(size)
        self._prev_layer: Layer = prev_layer
        self._weights: npt.NDArray[np.float64] = np.random.rand(prev_layer.size, size)
        self._values: npt.NDArray[np.float64] = np.zeros(size)
        self._activation_function: ActivationFunction = activation_function
