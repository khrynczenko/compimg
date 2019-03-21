"""compimg exceptions module"""
import numpy as np

from typing import Sequence


class DifferentShapesError(Exception):
    def __init__(self, shape1: Sequence[int], shape2: Sequence[int]):
        super().__init__(
            f"Images have different shapes: {shape1} != {shape2}.")


class DifferentDTypesError(Exception):
    def __init__(self, dtype1: np.dtype, dtype2: np.dtype):
        super().__init__(
            f"Images have different dtypes: {dtype1.name} != {dtype2.name}.")


class NegativePadAmountError(Exception):
    def __init__(self, amount):
        super().__init__(
            f"Pad cannot be negative value like {amount}.")
