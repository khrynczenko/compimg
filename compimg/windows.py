"""
Module with SlidingWindow interface and its implementations.
"""

import abc
import itertools
import numpy as np

from typing import Generator, Tuple
from compimg import kernels

Rows = int
Columns = int


class SlidingWindow(abc.ABC):
    @abc.abstractmethod
    def slide(self, image: np.ndarray) -> Generator[np.ndarray, None, None]:
        """
        Using some windows slides over image returning its changed/unchanged
        fragments.

        :param image: Image to slide over.
        :return: Generator that returns views returned by window.
        """


class IdentitySlidingWindow(SlidingWindow):
    """
    Slides through the image without making any changes.

    """

    def __init__(self, shape: Tuple[Rows, Columns],
                 stride: Tuple[Rows, Columns]):
        self._shape = shape
        self._stride = stride

    def slide(self, image: np.ndarray) -> Generator[np.ndarray, None, None]:
        starting_rows_range = range(0, image.shape[0], self._stride[0])
        starting_columns_range = range(0, image.shape[1], self._stride[1])
        starting_row_indices = itertools.takewhile(
            lambda index: index + self._shape[0] <= image.shape[0],
            starting_rows_range
        )
        starting_column_indices = itertools.takewhile(
            lambda index: index + self._shape[1] <= image.shape[1],
            starting_columns_range
        )
        for i, j in itertools.product(starting_row_indices,
                                      starting_column_indices):
            yield image[i:i + self._shape[0], j:j + self._shape[1]]


class GaussianSlidingWindow(SlidingWindow):
    def __init__(self, shape: Tuple[Rows, Columns],
                 stride: Tuple[Rows, Columns],
                 sigma: float):
        self._shape = shape
        self._stride = stride
        self._gaussian_window = kernels.make_guassian_kernel(shape, sigma)

    def slide(self, image: np.ndarray) -> Generator[np.ndarray, None, None]:
        gaussian_window = self._gaussian_window
        if image.ndim == 3:
            gaussian_window = kernels._replicate(gaussian_window, 3)
        return (gaussian_window * window for window in
                IdentitySlidingWindow(self._shape, self._stride).slide(image))
