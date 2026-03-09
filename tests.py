from unittest import TestCase

import numpy as np
from numpy.typing import NDArray

from activation_functions import ReLU
from layers import DenseLayer
from loss_functions import MeanSquaredError
from network import Network


class Tests(TestCase):
    def test_add(self):
        np.random.seed(0)

        train_data = np.zeros((1000, 3))
        for i in range(1000):
            # TODO: Support feature scaling to better handle large numbers
            a, b = np.random.randint(-100, 100, size=2)
            train_data[i] = [a, b, a + b]

        x_train = train_data[:800, :2]
        y_train = train_data[:800, 2:]

        network = Network(input_size=2)
        network.add_dense_layer(size=1)
        network.train(
            x_train,
            y_train,
            batch_size=32,
            epochs=10,
            learning_rate=0.0001,
            loss_function=MeanSquaredError(),
        )

        x_test = train_data[800:, :2]
        y_test = train_data[800:, 2:]
        predicted = network.predict(x_test)
        max_difference = np.abs(y_test - predicted).max()
        self.assertLess(max_difference, 0.01)
