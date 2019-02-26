"""compimg exceptions module"""
import numpy as np


class DifferentShapesError(Exception):
    def __init__(self, shape1, shape2):
        super().__init__(f"Images have different shapes: {shape1} != {shape2}.")


class DifferentDTypesError(Exception):
    def __init__(self, dtype1: np.dtype, dtype2: np.dtype):
        super().__init__(f"Images have different dtypes: {dtype1.name} != {dtype2.name}.")
