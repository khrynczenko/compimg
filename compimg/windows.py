"""
Module with SlidingWindow interface and its implementations.
"""

import abc
import itertools
import numpy as np

from typing import Generator, Tuple

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

    def __init__(
        self, shape: Tuple[Rows, Columns], stride: Tuple[Rows, Columns]
    ):
        self._shape = shape
        self._stride = stride

    def slide(self, image: np.ndarray) -> Generator[np.ndarray, None, None]:
        starting_rows_range = range(0, image.shape[0], self._stride[0])
        starting_columns_range = range(0, image.shape[1], self._stride[1])
        starting_row_indices = itertools.takewhile(
            lambda index: index + self._shape[0] <= image.shape[0],
            starting_rows_range,
        )
        starting_column_indices = itertools.takewhile(
            lambda index: index + self._shape[1] <= image.shape[1],
            starting_columns_range,
        )
        return (
            image[i : i + self._shape[0], j : j + self._shape[1]]
            for i, j in itertools.product(
                starting_row_indices, starting_column_indices
            )
        )
