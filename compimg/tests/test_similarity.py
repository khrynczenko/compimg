import numpy as np
import pytest

from compimg import similarity


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


class TestMSE:
    def test_comapre_returns_correct_result(self, image, reference_image):
        value = similarity.MSE().compare(image, reference_image)
        assert round(value, 2) == 1.33

    def test_compare_return_zero_when_identical_images(self, reference_image):
        value = similarity.MSE().compare(reference_image, reference_image)
        assert value == 0.0


class TestPSNR:
    def test_compare_returns_correct_result(self, image, reference_image):
        value = similarity.PSNR().compare(image, reference_image)
        assert round(value, 2) == 46.88

    def test_compare_returns_inf_if_images_are_identical(self,
                                                         reference_image):
        value = similarity.PSNR().compare(reference_image, reference_image)
        assert round(value, 2) == float("inf")


class TestSSIM:
    def test_compare_returns_one_when_images_are_identical(self):
        reference_image = np.ones((10, 10))
        value = similarity.SSIM().compare(reference_image, reference_image)
        assert value == 1.0

    def test_compare_returns_minus_one_when_images_are_completely_different(
            self):
        image = np.full((10, 10), fill_value=255, dtype=np.uint8)
        reference_image = np.zeros((10, 10), dtype=np.uint8)
        value = similarity.SSIM().compare(image, reference_image)
        assert round(value, 2) == 0.00
