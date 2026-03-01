class ReLU():
    def activate(self, value: float) -> float:
        return value if value > 0 else 0

    def slope(self, value: float) -> float:
        return 1 if value > 0 else 0
