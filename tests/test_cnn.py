from unittest import TestCase

import numpy as np
from sklearn.datasets import load_digits  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore
from sklearn.preprocessing import OneHotEncoder  # type: ignore

from neural_network_library.activation_functions import (
    ReLU,
    Sigmoid,
)
from neural_network_library.cnn_layers import (
    Conv2DLayer,
    FlattenLayer,
    GAPLayer,
    MaxPoolingLayer,
)
from neural_network_library.layers import DenseLayer, DropoutLayer
from neural_network_library.loss_functions import MeanSquaredError
from neural_network_library.network import Network


class Tests(TestCase):
    def test_digits(self):
        np.random.seed(0)

        digits = load_digits()
        x = digits.images.reshape(-1, 1, 8, 8).astype(np.float64)
        x /= 16.0
        y = digits.target.reshape(-1, 1)

        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.2, random_state=0
        )

        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        y_train = encoder.fit_transform(y_train)
        y_test = encoder.transform(y_test)

        network = Network(
            layers=[
                Conv2DLayer(16),
                Conv2DLayer(32),
                MaxPoolingLayer(2),
                FlattenLayer(),
                DropoutLayer(),
                DenseLayer(size=128, activation_function=ReLU()),
                DenseLayer(size=64, activation_function=ReLU()),
                DenseLayer(size=10),
            ]
        )
        network.train(
            x_train,
            y_train,
            batch_size=32,
            epochs=20,
            learning_rate=0.1,
            loss_function=MeanSquaredError(),
        )
        predicted = network.predict(x_test)

        y_predicted_categories = np.argmax(predicted, axis=1)
        y_actual_categories = np.argmax(y_test, axis=1)

        accuracy = np.mean(y_predicted_categories == y_actual_categories)
        self.assertGreater(accuracy, 0.95)
