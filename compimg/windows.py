"""
Module with SlidingWindow interface and its implementations.
"""

import abc
import itertools
import numpy as np

from typing import Generator, Tuple
from compimg._internals import _utilities
from compimg.pads import Pad, ConstantPad

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


class KernelApplyingSlidingWindow(SlidingWindow):

    def __init__(self, kernel: np.ndarray,
                 pad: Pad = ConstantPad(0, 1)):
        self._kernel = kernel
        self._pad = pad

    def slide(self, image: np.ndarray) -> Generator[np.ndarray, None, None]:
        original_dtype = image.dtype
        image = image.astype(np.float64)
        kernel = self._kernel.astype(np.float64)
        if image.ndim == 3 and kernel.ndim == 2:
            kernel = self._replicate(kernel, 3)
        slider = IdentitySlidingWindow(kernel.shape[:2], (1, 1))
        filtered_image = self._pad.apply(image)
        min, max = _utilities.get_dtype_range(original_dtype)
        return (np.sum(slide * kernel).clip(min, max).astype(original_dtype)
                for slide in
                slider.slide(filtered_image))

    def _replicate(self, array: np.ndarray, dim: int) -> np.ndarray:
        new = np.zeros((array.shape[0], array.shape[1], dim),
                       dtype=array.dtype)
        for i, j in itertools.product(range(new.shape[0]),
                                      range(new.shape[1])):
            new[i, j] = np.repeat(array[i, j], dim)
        return new
