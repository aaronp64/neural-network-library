import numpy as np
from sklearn.datasets import load_iris  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # type: ignore

from neural_network_library.activation_functions import (
    ReLU,
    Sigmoid,
)
from neural_network_library.layers import DenseLayer
from neural_network_library.loss_functions import MeanSquaredError
from neural_network_library.network import Network

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
    epochs=25,
    learning_rate=0.1,
    loss_function=MeanSquaredError(),
    show_progress=True,
)
predicted = network.predict(x_test)

y_predicted_categories = np.argmax(predicted, axis=1)
y_actual_categories = np.argmax(y_test, axis=1)

accuracy = np.mean(y_predicted_categories == y_actual_categories)
print(f"Accuracy: {accuracy*100:.2f}%")
