from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

class LossFunction(ABC):
    @abstractmethod
    def apply(
            self, actual: NDArray[np.float64],
            predicted: NDArray[np.float64]) -> np.float64:
        pass

    @abstractmethod
    def derivative(
            self, actual: NDArray[np.float64],
            predicted: NDArray[np.float64]) -> NDArray[np.float64]:
        pass

class MeanSquaredError(LossFunction):
    def apply(
            self, actual: NDArray[np.float64],
            predicted: NDArray[np.float64]) -> np.float64:
        squared: NDArray[np.float64] = np.square(predicted - actual)
        batch_totals: NDArray[np.float64] = np.sum(squared, axis=1)
        return np.float64(np.mean(batch_totals))

    def derivative(
            self, actual: NDArray[np.float64],
            predicted: NDArray[np.float64]) -> NDArray[np.float64]:
        return (2 / actual.shape[0]) * (predicted - actual)
