from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt

class LossFunction(ABC):
    @abstractmethod
    def loss(
            self, actual: npt.NDArray[np.float64],
            predicted: npt.NDArray[np.float64]) -> np.float64:
        pass

    @abstractmethod
    def gradient(
            self, actual: npt.NDArray[np.float64],
            predicted: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        pass

def MeanSquaredError(LossFunction):
    def loss(
            self, actual: npt.NDArray[np.float64],
            predicted: npt.NDArray[np.float64]) -> np.float64:
        return np.float64(np.mean((predicted - actual) ** 2))

    def gradient(
            self, actual: npt.NDArray[np.float64],
            predicted: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return (2 / actual.size) * (predicted - actual)
