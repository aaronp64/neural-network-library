"""
Provides Network class for neural network.
"""

import numpy as np

from .layers import Layer
from .loss_functions import LossFunction


class Network:
    """
    A neural network class for connecting multiple layers and handling training/prediction.

    Args:
        layers (list[Layer]): Layers to be used in this network.
    """

    def __init__(self, layers: list[Layer]):
        self._layers: list[Layer] = layers

    def predict(self, input_data: np.ndarray) -> np.ndarray:
        """
        Passes input through layers to predict output values.

        Args:
            input_data (np.ndarray): Input to use for prediction.

        Returns:
            np.ndarray: Output of final layer.
        """
        output = input_data
        for layer in self._layers:
            output = layer.forward_pass(output)
        return output

    def train(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        batch_size: int,
        epochs: int,
        learning_rate: float,
        loss_function: LossFunction,
    ) -> None:
        """
        Trains the network by predicting output, calculating loss, and backpropagation.

        Args:
            x_train (np.ndarray): Input data.
            y_train (np.ndarray): Actual/Expected output data.
            batch_size (int): Number of rows to use in each pass.
            epochs (int): Number of times to repeat through data.
            learning_rate (float): How much to adjust weights/biases in response to gradient.
            loss_function (LossFunction): Loss function to compare predictions with y_train.
        """
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
