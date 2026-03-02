from dataclasses import dataclass

@dataclass(slots=True)
class Synapse:
    from_index: int
    to_index: int
    weight: float
