import numpy as np
from numpy.typing import NDArray

from activation_functions import ActivationFunction, Linear
from layers import DenseLayer, Layer
from loss_functions import LossFunction


class Network:
    def __init__(self, input_size: int):
        self._input_size: int = input_size
        self._layers: list[Layer] = []

    def add_dense_layer(
        self, size: int, activation_function: ActivationFunction = Linear()
    ) -> None:
        input_size: int = self._layers[-1].size if self._layers else self._input_size
        self._layers.append(DenseLayer(input_size, size, activation_function))

    def predict(self, input_data: NDArray[np.float64]) -> NDArray[np.float64]:
        self._validate_input_size(input_data)

        output = input_data
        for layer in self._layers:
            output = layer.forward_pass(output)
        return output

    def train(
        self,
        x_train: NDArray[np.float64],
        y_train: NDArray[np.float64],
        batch_size: int,
        epochs: int,
        learning_rate: float,
        loss_function: LossFunction,
    ) -> None:
        self._validate_input_size(x_train)
        training_size: int = x_train.shape[0]

        for epoch in range(epochs):
            # TODO: Print loss/accuracy for each epoch
            # TODO: Early stopping with message
            for i in range(0, training_size, batch_size):
                x_batch = x_train[i : i + batch_size]
                y_batch = y_train[i : i + batch_size]

                output = self.predict(x_batch)
                gradient = loss_function.derivative(y_batch, output)

                for layer in reversed(self._layers):
                    gradient = layer.backward_pass(gradient, learning_rate)

    def _validate_input_size(self, input_data: NDArray[np.float64]) -> None:
        if input_data.shape[-1] != self._input_size:
            raise ValueError(
                f"Invalid input size, expected {self._input_size} features"
            )
