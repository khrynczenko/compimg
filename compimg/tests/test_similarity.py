import numpy as np
import pytest

from compimg.similarity import MSE, PSNR, SSIM
from compimg.exceptions import DifferentDTypesError, DifferentShapesError


@pytest.fixture
def reference_image():
    return np.array([
        [1, 2, 3],
        [4, 5, 6]
    ], dtype=np.uint8)


@pytest.fixture
def image():
    return np.array([
        [3, 2, 1],
        [4, 5, 6]
    ], dtype=np.uint8)


@pytest.mark.parametrize("metric", [MSE(), PSNR(), SSIM()])
def test_if_different_shapes_guard_raises(metric):
    wrong_shape_x = np.zeros((10, 10, 2))
    wrong_shape_y = np.zeros((20, 20, 2))
    with pytest.raises(DifferentShapesError):
        metric.compare(wrong_shape_x, wrong_shape_y)


@pytest.mark.parametrize("metric", [MSE(), PSNR(), SSIM()])
def test_if_different_dtypes_guard_raises(metric):
    wrong_dtype_x = np.zeros((10, 10, 2), dtype=np.float32)
    wrong_dtype_y = np.zeros((10, 10, 2), dtype=np.uint8)
    with pytest.raises(DifferentDTypesError):
        metric.compare(wrong_dtype_x, wrong_dtype_y)


class TestMSE:
    def test_compare_returns_correct_result(self, image, reference_image):
        value = MSE().compare(image, reference_image)
        assert round(value, 2) == 1.33

    def test_compare_returns_zero_when_identical_images(self, reference_image):
        value = MSE().compare(reference_image, reference_image)
        assert value == 0.0


class TestPSNR:
    def test_compare_returns_correct_result(self, image, reference_image):
        value = PSNR().compare(image, reference_image)
        assert round(value, 2) == 46.88

    def test_compare_returns_inf_if_images_are_identical(self,
                                                         reference_image):
        value = PSNR().compare(reference_image, reference_image)
        assert round(value, 2) == float("inf")


class TestSSIM:
    def test_compare_returns_one_when_images_are_identical(self):
        reference_image = np.ones((10, 10, 3))
        value = SSIM().compare(reference_image, reference_image)
        assert value == 1.0

    def test_compare_returns_minus_one_when_images_are_completely_different(
            self):
        image = np.full((10, 10, 3), fill_value=255, dtype=np.uint8)
        reference_image = np.zeros((10, 10, 3), dtype=np.uint8)
        value = SSIM().compare(image, reference_image)
        assert round(value, 2) == 0.00
