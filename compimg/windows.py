"""
Module with SlidingWindow interface and its implementations.
"""

import abc
import itertools
import numpy as np

from numbers import Number
from typing import Generator, Tuple
from compimg._internals import _utilities


class SlidingWindow(abc.ABC):
    @abc.abstractmethod
    def slide(self, image: np.ndarray) -> Generator[np.ndarray, None, None]:
        """
        Should return fragments of a given image.

        """


Rows = int
Columns = int


class DefaultSlidingWindow(SlidingWindow):
    def __init__(self, size: Tuple[Rows, Columns],
                 stride: Tuple[Rows, Columns]):
        self._size = size
        self._stride = stride

    def slide(self, image: np.ndarray) -> Generator[np.ndarray, None, None]:
        starting_rows_range = range(0, image.shape[0], self._stride[0])
        starting_columns_range = range(0, image.shape[1], self._stride[1])
        starting_row_indices = itertools.takewhile(
            lambda index: index + self._size[0] <= image.shape[0],
            starting_rows_range
        )
        starting_column_indices = itertools.takewhile(
            lambda index: index + self._size[1] <= image.shape[1],
            starting_columns_range
        )
        for i, j in itertools.product(starting_row_indices,
                                      starting_column_indices):
            yield image[i:i + self._size[0], j:j + self._size[1]]


class BorderSolution(abc.ABC):
    """
    When performing convolution one needs to decide what to do filter is near
    border(s). Instances implementing this class address that problem.
    """

    @abc.abstractmethod
    def apply(self, image: np.ndarray) -> np.ndarray:
        """Applies solution"""


class Pad(BorderSolution):
    """
    Adds rows/columns of zeros at the edges of an image.
    """

    def __init__(self, value: Number, amount: int):
        """
        :param value: Value to pad with.
        :param amount: Amount of rows/columns to be added.
        """
        self._value = value
        self._amount = amount

    def apply(self, image: np.ndarray) -> np.ndarray:
        image_shape_with_zero_border = list(image.shape)
        image_shape_with_zero_border[0] = image_shape_with_zero_border[0] + 2
        image_shape_with_zero_border[1] = image_shape_with_zero_border[1] + 2
        zero_pad = np.full(image_shape_with_zero_border, self._value,
                           dtype=image.dtype)
        zero_pad[1:-1, 1:-1] = image
        return zero_pad


class KernelApplyingSlidingWindow(SlidingWindow):

    def __init__(self, kernel: np.ndarray,
                 border_solution: BorderSolution = Pad(0, 1)):
        self._kernel = kernel
        self._border_solution = border_solution

    def slide(self, image: np.ndarray) -> Generator[np.ndarray, None, None]:
        original_dtype = image.dtype
        image = image.astype(np.float64)
        kernel = self._kernel.astype(np.float64)
        if image.ndim == 3 and kernel.ndim == 2:
            kernel = self._replicate(kernel, 3)
        slider = DefaultSlidingWindow(kernel.shape[:2], (1, 1))
        filtered_image = self._border_solution.apply(image)
        min, max = _utilities.get_dtype_range(original_dtype)
        return (np.sum((slide.ravel() * kernel.ravel())).clip(min, max).astype(
            original_dtype)
            for slide in
            slider.slide(filtered_image))

    def _replicate(self, array: np.ndarray, dim: int) -> np.ndarray:
        new = np.zeros((array.shape[0], array.shape[1], dim),
                       dtype=array.dtype)
        for i, j in itertools.product(range(new.shape[0]),
                                      range(new.shape[1])):
            new[i, j] = np.repeat(array[i, j], dim)
        return new
