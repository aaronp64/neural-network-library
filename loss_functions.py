from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt

class LossFunction(ABC):
    @abstractmethod
    def apply(
            self, actual: npt.NDArray[np.float64],
            predicted: npt.NDArray[np.float64]) -> np.float64:
        pass

    @abstractmethod
    def derivative(
            self, actual: npt.NDArray[np.float64],
            predicted: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        pass

class MeanSquaredError(LossFunction):
    def apply(
            self, actual: npt.NDArray[np.float64],
            predicted: npt.NDArray[np.float64]) -> np.float64:
        squared: npt.NDArray[np.float64] = np.square(predicted - actual)
        batch_totals: npt.NDArray[np.float64] = np.sum(squared, axis=1)
        return np.float64(np.mean(batch_totals))

    def derivative(
            self, actual: npt.NDArray[np.float64],
            predicted: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return (2 / actual.shape[0]) * (predicted - actual)
