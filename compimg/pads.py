"""
This module defines means to apply padding to images.

"""
import abc
import functools
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
        new_image_shape = list(image.shape)
        new_image_shape[0] = new_image_shape[0] + (self._amount * 2)
        new_image_shape[1] = new_image_shape[1] + (self._amount * 2)
        padded_image = np.full(new_image_shape,
                               self._value,
                               dtype=image.dtype)
        start = self._amount
        end = -self._amount
        padded_image[start:end, start: end] = image
        return padded_image


class EdgePad(Pad):
    """
    Replicates neighbouring pixels at edges.

    """

    @_raise_if_pad_amount_is_negative
    def __init__(self, amount: int):
        """

        :param value: Value to pad with (New edges will be filled with that
        value.
        :param amount: Amount of rows/columns to be added.
        :raises: When amount is negative.
        """
        self._amount = amount
        self._func = functools.partial(np.pad, pad_width=(
            (self._amount, self._amount),
            (self._amount, self._amount)), mode="edge")
        if self._amount == 0:
            self.apply = lambda x: x

    def apply(self, image: np.ndarray) -> np.ndarray:
        return self._func(image)
