from unittest import TestCase

import numpy as np

from neural_network_library.activation_functions import (
    LeakyReLU,
    Linear,
    ReLU,
    Sigmoid,
    Tanh,
)


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
