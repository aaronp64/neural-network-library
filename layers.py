from collections.abc import Callable

class DenseLayer():
    def __init__(self, size: int, activation_func: Callable[[float], float]) -> None:
        self._size: int = size
        self._connections: list[tuple[int, int, float]] = []
        self._activation_func: Callable[[float], float] = activation_func

    def connect(self) -> None:
        pass
