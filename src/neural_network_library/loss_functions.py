"""
Provides loss function classes for a neural network.
"""

from abc import ABC, abstractmethod
from typing import override

import numpy as np


class LossFunction(ABC):
    """
    Abstract base class for loss functions.
    """

    @abstractmethod
    def apply(self, *, actual: np.ndarray, predicted: np.ndarray) -> np.float64:
        """
        Calculates the loss based on actual and predicted values.

        Args:
            actual (np.ndarray): Ground truth values.
            predicted (np.ndarray): Values predicted by the network.

        Returns:
            float: Calculated loss value.
        """

    @abstractmethod
    def derivative(self, *, actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
        """
        Calculates the gradient of loss based on actual and predicted values.

        Args:
            actual (np.ndarray): Ground truth values.
            predicted (np.ndarray): Values predicted by the network.

        Returns:
            np.ndarray: Calculated gradient of loss.
        """


class MeanSquaredError(LossFunction):
    """
    Mean Squared Error loss function.

    Calculates the average (across rows) of squared predicted/actual differences.
    """

    @override
    def apply(self, *, actual: np.ndarray, predicted: np.ndarray) -> np.float64:
        squared: np.ndarray = np.square(predicted - actual)
        row_totals: np.ndarray = np.sum(squared, axis=1)
        return np.float64(np.mean(row_totals))

    @override
    def derivative(self, *, actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
        return (2 / actual.shape[0]) * (predicted - actual)
