from activation_functions import ActivationFunction
from layers import DenseLayer, Layer


class Network:
    def __init__(self, input_size: int):
        self._input_size: int = input_size
        self._layers: list[Layer] = []

    def add_dense_layer(
        self, size: int, activation_function: ActivationFunction
    ) -> None:
        input_size: int = self._layers[-1].size if self._layers else self._input_size
        self._layers.append(DenseLayer(input_size, size, activation_function))
