from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt

class ActivationFunction(ABC):
    @abstractmethod
    def activate(self, values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        pass

    @abstractmethod
    def slope(self, values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        pass

class ReLU(ActivationFunction):
    def activate(self, values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return np.maximum(0, values)

    def slope(self, values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return (values > 0).astype(np.float64)
