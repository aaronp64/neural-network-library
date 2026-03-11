A small neural network library built in python.  My main goal with this project is to get a better understanding of how neural networks function by working through the implementation.

To install module/dependencies, run `pip install -e .` in the project directory.  Dependencies include numpy for matrix math and scikit-learn for preprocessing data. For development (formatting/linting/type hint tools), use `pip install -e '.[dev]'`, and run tools using:

```
black .
mypy src
pylint src
```

Tests can be run using `python -m unittest` (may need to use `python3` instead of `python`).

Networks can be initialized, trained, and used for prediction using code like below (see `examples/` directory for more complete example usage):

```python
network = Network(
    layers=[
        DenseLayer(size=8, activation_function=ReLU()),
        DenseLayer(size=1), # Output layer
    ]
)
network.train(
    x_train,
    y_train,
    batch_size=32,
    epochs=50,
    learning_rate=0.01,
    loss_function=MeanSquaredError(),
)
predicted = network.predict(x_test)
```
