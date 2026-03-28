"""
Provides layers for Convolutional Neural Networks
"""

from typing import override

import numpy as np

from .activation_functions import ActivationFunction, ReLU
from .layers import Layer


class Conv2DLayer(Layer):
    """
    Identifies features in input images using sliding filters.

    Currently uses fixed padding of 1, kernel size of 3x3, and stride of 1.

    Args:
        size (int): The number of filters/kernels to use.
        activation_function (ActivationFunction): Activation function to apply to output,
            defaults to ReLU.
    """

    def __init__(
        self, size: int, activation_function: ActivationFunction = ReLU()
    ) -> None:
        # TODO: Allow passing in kernel/padding size?
        self._size: int = size
        self._activation_function: ActivationFunction = activation_function

        self._weights: np.ndarray | None = None
        self._biases: np.ndarray = np.zeros(self._size)
        self._input_cache: np.ndarray | None = None
        self._z_cache: np.ndarray | None = None  # Pre-activation output

    @override
    def forward(self, input_data: np.ndarray, *, is_training: bool) -> np.ndarray:
        batch_size, channels, height, width = input_data.shape
        if self._weights is None:
            # TODO: Allow passing in different initialization methods
            self._weights = np.random.randn(self._size, channels, 3, 3) * np.sqrt(
                2 / (channels * 3 * 3)
            )

        self._input_cache = input_data
        padded = np.pad(input_data, ((0, 0), (0, 0), (1, 1), (1, 1)))

        self._z_cache = np.zeros((batch_size, self._size, height, width))

        # TODO: im2col optimization
        for h in range(height):
            for w in range(width):
                view = padded[:, :, h : h + 3, w : w + 3]
                self._z_cache[:, :, h, w] = (
                    np.tensordot(view, self._weights, axes=((1, 2, 3), (1, 2, 3)))
                    + self._biases
                )

        return self._activation_function.apply(self._z_cache)

    @override
    def backward(
        self, output_gradient: np.ndarray, *, learning_rate: float
    ) -> np.ndarray:
        assert self._weights is not None
        assert self._input_cache is not None
        assert self._z_cache is not None

        z_gradient: np.ndarray = output_gradient * self._activation_function.derivative(
            self._z_cache
        )
        _batch_size, _size, height, width = z_gradient.shape

        weight_gradient: np.ndarray = np.zeros_like(self._weights)
        bias_gradient: np.ndarray = np.zeros_like(self._biases)

        padded = np.pad(self._input_cache, ((0, 0), (0, 0), (1, 1), (1, 1)))
        padded_gradient: np.ndarray = np.zeros_like(padded)

        for h in range(height):
            for w in range(width):
                pos_gradient: np.ndarray = z_gradient[:, :, h, w]
                view = padded[:, :, h : h + 3, w : w + 3]
                weight_gradient += np.tensordot(pos_gradient, view, axes=(0, 0))
                bias_gradient += np.sum(pos_gradient, axis=0)
                padded_gradient[:, :, h : h + 3, w : w + 3] += np.tensordot(
                    pos_gradient, self._weights, axes=(1, 0)
                )

        self._weights -= learning_rate * weight_gradient / (width * height)
        self._biases -= learning_rate * bias_gradient / (width * height)
        input_gradient = padded_gradient[:, :, 1:-1, 1:-1]
        return input_gradient


class MaxPoolingLayer(Layer):
    """
    Downsamples input by taking max value from each region.

    Args:
        size (int): Height and width of square region.
    """

    def __init__(self, size: int = 2) -> None:
        # TODO: More customization:
        # - Separate pool_size/stride
        # - Separate height/width dimensions for each
        self._size: int = size
        self._input_cache: np.ndarray | None = None
        self._max_cache: np.ndarray | None = None

    @override
    def forward(self, input_data: np.ndarray, *, is_training: bool) -> np.ndarray:
        self._input_cache = input_data
        batch_size, channels, height, width = input_data.shape

        # TODO: Handle last row/column for odd sizes?
        out_h: int = (height - self._size) // self._size + 1
        out_w: int = (width - self._size) // self._size + 1
        output: np.ndarray = np.zeros((batch_size, channels, out_h, out_w))
        self._max_cache = np.zeros_like(input_data)

        for h in range(out_h):
            h_start: int = h * self._size
            h_end: int = h_start + self._size
            for w in range(out_w):
                w_start: int = w * self._size
                w_end: int = w_start + self._size

                view: np.ndarray = input_data[:, :, h_start:h_end, w_start:w_end]
                max_vals: np.ndarray = np.max(view, axis=(2, 3))
                output[:, :, h, w] = max_vals

                max_positions = view == max_vals[:, :, np.newaxis, np.newaxis]
                max_count = np.sum(max_positions, axis=(2, 3))

                self._max_cache[:, :, h_start:h_end, w_start:w_end] += (
                    max_positions / max_count[:, :, np.newaxis, np.newaxis]
                )

        return output

    @override
    def backward(
        self, output_gradient: np.ndarray, *, learning_rate: float
    ) -> np.ndarray:
        assert self._input_cache is not None
        assert self._max_cache is not None

        out_h, out_w = output_gradient.shape[2], output_gradient.shape[3]

        input_gradient = np.zeros_like(self._input_cache)
        for h in range(out_h):
            h_start: int = h * self._size
            h_end: int = h_start + self._size
            for w in range(out_w):
                w_start: int = w * self._size
                w_end: int = w_start + self._size

                view_gradient: np.ndarray = output_gradient[:, :, h, w][
                    :, :, np.newaxis, np.newaxis
                ]
                input_gradient[:, :, h_start:h_end, w_start:w_end] += (
                    view_gradient * self._max_cache[:, :, h_start:h_end, w_start:w_end]
                )

        return input_gradient


class FlattenLayer(Layer):
    """
    Reshapes input to flatten non-batch dimensions into one dimension.
    """

    def __init__(self) -> None:
        self._input_shape: tuple[int, int, int, int] | None = None

    @override
    def forward(self, input_data: np.ndarray, *, is_training: bool) -> np.ndarray:
        self._input_shape = input_data.shape
        return input_data.reshape(self._input_shape[0], -1)

    @override
    def backward(
        self, output_gradient: np.ndarray, *, learning_rate: float
    ) -> np.ndarray:
        return output_gradient.reshape(self._input_shape)


class GAPLayer(Layer):
    """
    Takes average of feature maps, flattening height/width dimensions into single value.
    """

    def __init__(self) -> None:
        self._input_shape: tuple[int, int, int, int] | None = None

    @override
    def forward(self, input_data: np.ndarray, *, is_training: bool) -> np.ndarray:
        self._input_shape = input_data.shape
        return np.mean(input_data, axis=(2, 3))

    @override
    def backward(
        self, output_gradient: np.ndarray, *, learning_rate: float
    ) -> np.ndarray:
        assert self._input_shape is not None

        _batch_size, _channels, width, height = self._input_shape

        return np.broadcast_to(
            output_gradient[:, :, np.newaxis, np.newaxis], self._input_shape
        ) / (width * height)
