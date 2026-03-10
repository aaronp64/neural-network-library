from unittest import TestCase

import numpy as np

from activation_functions import LeakyReLU, Linear, ReLU, Sigmoid, Tanh
from layers import DenseLayer
from loss_functions import MeanSquaredError
from network import Network


class Tests(TestCase):
    def _test_activation_function(
        self, activation_function, input_data, activated_expected, derivative_expected
    ):
        activated = activation_function.apply(input_data)
        self.assertTrue(np.allclose(activated, activated_expected))

        derivative = activation_function.derivative(input_data)
        self.assertTrue(np.allclose(derivative, derivative_expected))

    def test_leaky_relu(self):
        self._test_activation_function(
            activation_function=LeakyReLU(0.1),
            input_data=np.array([-100.0, -1.0, 0.0, 1.0, 100.0]),
            activated_expected=np.array([-10.0, -0.1, 0.0, 1.0, 100.0]),
            derivative_expected=np.array([0.1, 0.1, 0.1, 1.0, 1.0]),
        )

    def test_linear(self):
        self._test_activation_function(
            activation_function=Linear(),
            input_data=np.array([-100.0, -1.0, 0.0, 1.0, 100.0]),
            activated_expected=np.array([-100.0, -1.0, 0.0, 1.0, 100.0]),
            derivative_expected=np.array([1.0, 1.0, 1.0, 1.0, 1.0]),
        )

    def test_relu(self):
        self._test_activation_function(
            activation_function=ReLU(),
            input_data=np.array([-100.0, -1.0, 0.0, 1.0, 100.0]),
            activated_expected=np.array([0.0, 0.0, 0.0, 1.0, 100.0]),
            derivative_expected=np.array([0.0, 0.0, 0.0, 1.0, 1.0]),
        )

    def test_sigmoid(self):
        self._test_activation_function(
            activation_function=Sigmoid(),
            input_data=np.array([-100.0, -1.0, 0.0, 1.0, 100.0]),
            activated_expected=np.array([0.0, 0.26894, 0.5, 0.73106, 1.0]),
            derivative_expected=np.array([0.0, 0.19661, 0.25, 0.19661, 0.0]),
        )

    def test_tanh(self):
        self._test_activation_function(
            activation_function=Tanh(),
            input_data=np.array([-100.0, -1.0, 0.0, 1.0, 100.0]),
            activated_expected=np.array([-1.0, -0.76159, 0.0, 0.76159, 1.0]),
            derivative_expected=np.array([0.0, 0.419974, 1.0, 0.419974, 0.0]),
        )

    def test_add(self):
        np.random.seed(0)

        train_data = np.zeros((1000, 3))
        for i in range(1000):
            # TODO: Support feature scaling to better handle large numbers
            a, b = np.random.randint(-100, 100, size=2)
            train_data[i] = [a, b, a + b]

        x_train = train_data[:800, :2]
        y_train = train_data[:800, 2:]

        network = Network(
            layers=[
                DenseLayer(size=1),
            ]
        )
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
