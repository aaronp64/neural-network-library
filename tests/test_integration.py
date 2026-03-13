from unittest import TestCase

import numpy as np
from sklearn.datasets import load_iris, make_moons  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # type: ignore

from neural_network_library.activation_functions import (
    LeakyReLU,
    ReLU,
    Sigmoid,
    Tanh,
)
from neural_network_library.layers import DenseLayer, DropoutLayer
from neural_network_library.loss_functions import MeanSquaredError
from neural_network_library.network import Network


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

    def test_iris(self):
        np.random.seed(0)

        iris = load_iris()
        x, y = iris.data, iris.target.reshape(-1, 1)

        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x)

        encoder = OneHotEncoder(sparse_output=False)
        y_encoded = encoder.fit_transform(y)

        x_train, x_test, y_train, y_test = train_test_split(
            x_scaled, y_encoded, test_size=0.2, random_state=0
        )

        network = Network(
            layers=[
                DenseLayer(size=5, activation_function=ReLU()),
                DenseLayer(size=3, activation_function=Sigmoid()),
            ]
        )
        network.train(
            x_train,
            y_train,
            batch_size=4,
            epochs=100,
            learning_rate=0.1,
            loss_function=MeanSquaredError(),
        )
        predicted = network.predict(x_test)

        y_predicted_categories = np.argmax(predicted, axis=1)
        y_actual_categories = np.argmax(y_test, axis=1)

        accuracy = np.mean(y_predicted_categories == y_actual_categories)
        self.assertGreater(accuracy, 0.95)

    def test_moons(self):
        np.random.seed(0)

        x, y = make_moons(n_samples=5000, noise=0.2, random_state=0)
        y = y.reshape(-1, 1)

        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x)

        encoder = OneHotEncoder(sparse_output=False)
        y_encoded = encoder.fit_transform(y)

        x_train, x_test, y_train, y_test = train_test_split(
            x_scaled, y_encoded, test_size=0.2, random_state=0
        )

        network = Network(
            layers=[
                DenseLayer(size=128, activation_function=LeakyReLU()),
                DropoutLayer(),
                DenseLayer(size=64, activation_function=Tanh()),
                DropoutLayer(0.2),
                DenseLayer(size=32, activation_function=ReLU()),
                DenseLayer(size=2, activation_function=Sigmoid()),
            ]
        )
        network.train(
            x_train,
            y_train,
            batch_size=8,
            epochs=25,
            learning_rate=0.1,
            loss_function=MeanSquaredError(),
        )
        predicted = network.predict(x_test)

        y_predicted_categories = np.argmax(predicted, axis=1)
        y_actual_categories = np.argmax(y_test, axis=1)

        accuracy = np.mean(y_predicted_categories == y_actual_categories)
        self.assertGreater(accuracy, 0.95)
