from unittest import TestCase

import numpy as np

from neural_network_library.loss_functions import MeanSquaredError


class Tests(TestCase):
    def _test_loss_function(
        self, *, loss_function, actual, predicted, loss_expected, derivative_expected
    ):
        loss = loss_function.apply(actual=actual, predicted=predicted)
        self.assertTrue(np.isclose(loss, loss_expected))

        derivative = loss_function.derivative(actual=actual, predicted=predicted)
        self.assertTrue(np.allclose(derivative, derivative_expected))

    def test_mse(self):
        self._test_loss_function(
            loss_function=MeanSquaredError(),
            actual=np.array(
                [
                    [1, 2, 3],
                    [-1, 3, -2],
                ]
            ),
            predicted=np.array(
                [
                    [1, 1, 3],
                    [0, 0, 0],
                ]
            ),
            loss_expected=7.5,
            derivative_expected=np.array(
                [
                    [0, -1, 0],
                    [1, -3, 2],
                ]
            ),
        )
