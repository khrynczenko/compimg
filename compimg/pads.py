"""
This module defines means to apply padding to images.

"""
import abc
import numpy as np

from abc import ABC
from numbers import Number
from typing import Callable
from compimg._internals._decorators import _raise_if_pad_amount_is_negative


class Pad(ABC):
    """
    When performing convolution one needs to decide what to do filter is near
    border(s). Instances implementing this class address that problem.

    """

    @abc.abstractmethod
    def apply(self, image: np.ndarray) -> np.ndarray:
        """
        Pads given image.

        :param image: Image to pad.
        :return: Padded image.

        """


class NoPad(Pad):
    """
    Helper class when one has to pass Pad object but does not want apply
    any padding.

    """

    def apply(self, image: np.ndarray) -> np.ndarray:
        return image


class FromFunctionPad(Pad):
    def __init__(self, function: Callable[[np.ndarray], np.ndarray]):
        self._function = function

    def apply(self, image: np.ndarray) -> np.ndarray:
        """
        Pads given image.

        :param image: Image to pad.
        :return: Padded image.

        """
        return self._function(image)


class ConstantPad(Pad):
    """
    Adds rows/columns of chosen value at the edges of an image.

    """

    @_raise_if_pad_amount_is_negative
    def __init__(self, value: Number, amount: int):
        """

        :param value: Value to pad with (New edges will be filled with that
        value.
        :param amount: Amount of rows/columns to be added.
        :raises: When amount is negative.

        """
        self._value = value
        self._amount = amount
        if self._amount == 0:
            self.apply = lambda x: x

    def apply(self, image: np.ndarray) -> np.ndarray:
        pad_width = [[self._amount, self._amount],
                     [self._amount, self._amount]]
        if image.ndim == 3:
            pad_width.append([0, 0])
        return np.pad(image,
                      pad_width,
                      mode="constant",
                      constant_values=self._value)


class EdgePad(Pad):
    """
    Replicates neighbouring pixels at edges.

    """

    @_raise_if_pad_amount_is_negative
    def __init__(self, amount: int):
        """

        :param amount: Amount of rows/columns to be added.
        :raises: When amount is negative.

        """
        self._amount = amount
        if self._amount == 0:
            self.apply = lambda x: x

    def apply(self, image: np.ndarray) -> np.ndarray:
        pad_width = [[self._amount, self._amount],
                     [self._amount, self._amount]]
        if image.ndim == 3:
            pad_width.append([0, 0])
        return np.pad(image,
                      pad_width,
                      mode="edge")
