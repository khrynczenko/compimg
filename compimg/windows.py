"""
Module with SlidingWindow interface and its implementations.
"""

import abc
import itertools
import numpy as np

from typing import Generator, Tuple


class SlidingWindow(abc.ABC):
    @abc.abstractmethod
    def slide(self, image: np.ndarray) -> Generator[np.ndarray, None, None]:
        """
        Should return fragments of a given image.

        """


Rows, Columns = int, int


class DefaultSlidingWindow(SlidingWindow):
    def __init__(self, size: Tuple[Rows, Columns],
                 stride: Tuple[Rows, Columns]):
        self._size = size
        self._stride = stride

    def slide(self, image: np.ndarray) -> Generator[np.ndarray, None, None]:
        starting_row_indices = range(0, image.shape[0], self._stride[0])
        starting_column_indices = range(0, image.shape[1], self._stride[1])
        starting_row_indices = itertools.takewhile(
            lambda index: index + self._size[0] <= image.shape[0],
            starting_row_indices
        )
        starting_column_indices = itertools.takewhile(
            lambda index: index + self._size[1] <= image.shape[1],
            starting_column_indices
        )
        for i, j in itertools.product(starting_row_indices,
                                      starting_column_indices):
            yield image[i:i + self._size[0], j:j + self._size[1]]
