import numpy as np
import pytest

from compimg.exceptions import (KernelBiggerThanImageError,
                                KernelShapeNotOddError)
from compimg import kernels


def test_convolve_work_correctly_on_one_channel_image():
    image = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.uint8)
    kernel = kernels.BOX_BLUR_3X3

    filtered_image = kernels.convolve(image, kernel)
    assert filtered_image.shape == (1, 1)
    assert filtered_image[0][0] == 5.0


def test_convolve_work_correctly_on_three_channels():
    image = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.uint8)
    image = image.reshape((3, 3, 1))
    image = np.concatenate((image, image, image), axis=2)
    kernel = kernels.BOX_BLUR_3X3

    filtered_image = kernels.convolve(image, kernel)
    assert filtered_image.shape == (1, 1, 3)
    assert np.array_equal(filtered_image[0][0], [5.0, 5.0, 5.0])


def test_convolve_raises_when_kernel_is_bigger():
    image = np.zeros((2, 2))
    kernel = kernels.BOX_BLUR_3X3
    with pytest.raises(KernelBiggerThanImageError):
        kernels.convolve(image, kernel)


def test_convolve_raises_when_kernel_is_not_odd_shape():
    image = np.zeros((10, 10))
    kernel = np.array(np.ones((4, 4)))
    with pytest.raises(KernelShapeNotOddError):
        kernels.convolve(image, kernel)
