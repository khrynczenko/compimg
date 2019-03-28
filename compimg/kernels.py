"""Image processing using kernels."""
import itertools
import numpy as np

from typing import Tuple
from compimg.exceptions import KernelBiggerThanImageError
from compimg.windows import IdentitySlidingWindow
from compimg._internals import _utilities

BOX_BLUR_3X3: np.ndarray = np.full((3, 3), 1.0 / 9.0, dtype=np.float64)
BOX_BLUR_4X4: np.ndarray = np.full((4, 4), 1.0 / 16.0, dtype=np.float64)
BOX_BLUR_5X5: np.ndarray = np.full((4, 4), 1.0 / 25.0, dtype=np.float64)
GAUSSIAN_BLUR_3x3: np.ndarray = (1.0 / 16.0) * np.array(
    [[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float64)
VERTICAL_SOBEL_3x3: np.ndarray = np.array(
    [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
HORIZONTAL_SOBEL_3x3: np.ndarray = np.array(
    [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)


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

    """
    if kernel.shape[:2] > image.shape[:2]:
        raise KernelBiggerThanImageError(kernel.shape, image.shape)
    original_dtype = image.dtype
    image = image.astype(np.float64, copy=False)
    kernel = kernel.astype(np.float64, copy=False)
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


def make_guassian_kernel(shape: Tuple[int, int], sigma: float):
    """
    Produces Two-dimensional Gaussian function according to
    https://en.wikipedia.org/wiki/Gaussian_function.

    :param shape: Shape of the kernel.
    :param sigma: Sigma to use in the formula.
    :return: Gaussian kernel.
    """
    middle_x = shape[0] / 2
    middle_y = shape[1] / 2
    start = np.ones(shape, dtype=np.float64)
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
