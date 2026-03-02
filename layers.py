from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Self

from synapse import Synapse

class Layer(ABC):
    def __init__(self, size: int) -> None:
        self._size: int = size

    @property
    def size(self) -> int:
        return self._size

    @abstractmethod
    def connect(self, prev_layer: Self):
        pass

class InputLayer(Layer):
    def __init__(self, size: int) -> None:
        super().__init__(size)

    def connect(self, prev_layer: Layer) -> None:
        raise NotImplementedError("InputLayer should always be first layer in network.")

class DenseLayer(Layer):
    def __init__(self, size: int, activation_func: Callable[[float], float]) -> None:
        super().__init__(size)
        self._synapses: list[Synapse] = []
        self._activation_func: Callable[[float], float] = activation_func

    def connect(self, prev_layer: Layer) -> None:
        self._synapses = []
        for self_i in range(self.size):
            for prev_i in range(prev_layer.size):
                self._synapses.append(Synapse(prev_i, self_i, 0))
