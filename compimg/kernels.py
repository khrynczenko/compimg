"""
Image processing using kernels. Includes several ready to be used kernels
and convolution routines.

"""
import numpy as np
import compimg

from scipy import ndimage
from compimg.exceptions import (KernelBiggerThanImageError,
                                KernelShapeNotOddError,
                                KernelNot2DArray)

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
        Result :py:class:`numpy.ndarray` need to be processed properly before
        it can be used as an image again. For example one could divide its
        values by 255.0 and then cast its dtype to np.uint8.

    .. attention::
        In case when image has multiple channels kernel is going to be used
        separately for each image channel.

    :param image: Image on which to perform a convolution.
    :param kernel: Kernel to be used.
    :return: Convolved image (probably of different dtype).
    :raises KernelBiggerThanImageError: When kernel does not fit into image.
    :raises KernelShapeNotOddError: When kernel does not is of even shape.
    :raises KernelNot2DArray: When kernel is not a 2 dimensional array.
    """
    if kernel.ndim != 2:
        raise KernelNot2DArray(kernel.ndim)
    if kernel.shape[:2] > image.shape[:2]:
        raise KernelBiggerThanImageError(kernel.shape, image.shape)
    if kernel.shape[0] % 2 == 0 or kernel.shape[1] % 2 == 0:
        raise KernelShapeNotOddError(kernel.shape)
    output = np.zeros(image.shape, dtype=compimg.config.intermediate_dtype)
    if image.ndim > 2:
        for channel in range(image.shape[2]):
            output[:, :, channel] = (
                ndimage.convolve(image[:, :, channel],
                                 kernel,
                                 output=compimg.config.intermediate_dtype))
    else:
        output = ndimage.convolve(image,
                                  kernel,
                                  output=compimg.config.intermediate_dtype)

    output = output[kernel.shape[0] // 2: -(kernel.shape[0] // 2),
                    kernel.shape[1] // 2: -(kernel.shape[1] // 2)]
    return output
