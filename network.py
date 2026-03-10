import numpy as np
from numpy.typing import NDArray

from activation_functions import ActivationFunction, Linear
from layers import DenseLayer, Layer
from loss_functions import LossFunction


class Network:
    def __init__(self, layers: list[Layer]):
        self._layers: list[Layer] = layers

    def predict(self, input_data: NDArray[np.float64]) -> NDArray[np.float64]:
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
