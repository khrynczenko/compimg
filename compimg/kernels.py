"""Image processing using kernels."""
import itertools
import numpy as np
import compimg

from typing import Tuple
from compimg.exceptions import (KernelBiggerThanImageError,
                                KernelShapeNotOddError)
from compimg.windows import IdentitySlidingWindow
from compimg._internals import _utilities

BOX_BLUR_3X3: np.ndarray = np.full((3, 3), 1.0 / 9.0,
                                   dtype=compimg.config.intermediate_dtype)
BOX_BLUR_5X5: np.ndarray = np.full((5, 5), 1.0 / 25.0,
                                   dtype=compimg.config.intermediate_dtype)
GAUSSIAN_BLUR_3x3: np.ndarray = (1.0 / 16.0) * np.array(
    [[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=compimg.config.intermediate_dtype)
VERTICAL_SOBEL_3x3: np.ndarray = np.array(
    [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
    dtype=compimg.config.intermediate_dtype)
HORIZONTAL_SOBEL_3x3: np.ndarray = np.array(
    [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
    dtype=compimg.config.intermediate_dtype)


def convolve(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Performs the convolution using provided kernel.

    .. attention::
        In case when image has multiple channels and provided kernel has only
        one, the kernel values get replicated along every channel.

    :param image: Image on which to perform a convolution.
    :param kernel: Kernel to be used.
    :return: Convolved image.
    :raises KernelBiggerThanImageError: When kernel does not fit into image.
    :raises KernelShapeNotOddError: When kernel does not is of even shape.
    """
    if kernel.shape[:2] > image.shape[:2]:
        raise KernelBiggerThanImageError(kernel.shape, image.shape)
    if kernel.shape[0] % 2 == 0 or kernel.shape[1] % 2 == 0:
        raise KernelShapeNotOddError(kernel.shape)
    original_dtype = image.dtype
    image = image.astype(compimg.config.intermediate_dtype, copy=False)
    kernel = kernel.astype(compimg.config.intermediate_dtype, copy=False)
    axis = None
    if image.ndim == 3 and kernel.ndim == 2:  # Multichannel image
        kernel = _replicate(kernel, image.shape[2])
        axis = (0, 1)
    slider = IdentitySlidingWindow(kernel.shape[:2], (1, 1))
    min, max = _utilities._get_image_dtype_range(original_dtype)
    pixels = np.array(
        [np.sum(slide * kernel, axis=axis).clip(min, max).astype(
            original_dtype)
            for slide in
            slider.slide(image)])
    new_shape = list(image.shape)
    new_shape[0] = image.shape[0] - kernel.shape[0] + 1
    new_shape[1] = image.shape[1] - kernel.shape[1] + 1
    return pixels.reshape(new_shape)


def _unobtrusive_convolve(image: np.ndarray,
                          kernel: np.ndarray) -> np.ndarray:
    """
    Performs the convolution using provided kernel. Does not clip values
    change dtype or other things that are present in standard convolve.

    :param image: Image on which to perform a convolution.
    :param kernel: Kernel to be used.
    :return: Convolved image.
    :raises KernelBiggerThanImageError: When kernel does not fit into image.
    :raises KernelShapeNotOddError: When kernel does not is of even shape.
    """
    if kernel.shape[:2] > image.shape[:2]:
        raise KernelBiggerThanImageError(kernel.shape, image.shape)
    if kernel.shape[0] % 2 == 0 or kernel.shape[1] % 2 == 0:
        raise KernelShapeNotOddError(kernel.shape)
    axis = None
    if image.ndim == 3 and kernel.ndim == 2:  # Multichannel image
        kernel = _replicate(kernel, image.shape[2])
        axis = (0, 1)
    slider = IdentitySlidingWindow(kernel.shape[:2], (1, 1))
    pixels = np.asarray(
        [(slide * kernel).sum(axis=axis)
         for slide in
         slider.slide(image)], compimg.config.intermediate_dtype)
    new_shape = list(image.shape)
    new_shape[0] = image.shape[0] - kernel.shape[0] + 1
    new_shape[1] = image.shape[1] - kernel.shape[1] + 1
    return pixels.reshape(new_shape)


def make_guassian_kernel(shape: Tuple[int, int], sigma: float):
    """
    Produces two-dimensional Gaussian kernel according to
    https://en.wikipedia.org/wiki/Gaussian_function.

    :param shape: Shape of the kernel.
    :param sigma: Sigma to use in the formula.
    :return: Gaussian kernel.
    """
    middle_x = shape[0] / 2
    middle_y = shape[1] / 2
    start = np.ones(shape, dtype=compimg.config.intermediate_dtype)
    x_y = np.fromfunction(
        lambda i, j: (i - middle_x) ** 2 + (j - middle_y) ** 2, shape,
        dtype=np.float64)
    first = start / (sigma ** 2 * 2 * np.pi)
    second = first * (np.exp(-(x_y / (2 * sigma ** 2))))
    return second


def _replicate(array: np.ndarray, channels: int) -> np.ndarray:
    new = np.zeros((array.shape[0], array.shape[1], channels),
                   dtype=array.dtype)
    for i, j in itertools.product(range(new.shape[0]),
                                  range(new.shape[1])):
        new[i, j] = np.repeat(array[i, j], channels)
    return new
