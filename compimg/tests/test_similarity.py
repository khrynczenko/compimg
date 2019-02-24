import pytest
import numpy as np
from compimg import similarity


@pytest.fixture
def reference_image():
    return np.array([
        [1, 2, 3],
        [4, 5, 6]
    ], dtype=np.uint8)


@pytest.fixture
def target_image():
    return np.array([
        [3, 2, 1],
        [4, 5, 6]
    ], dtype=np.uint8)


class TestMSE:
    def test_comapre_returns_correct_result(self, reference_image, target_image):
        value = similarity.MSE().compare(target_image, reference_image)
        assert round(value, 2) == 1.33

    def test_compare_return_zero_when_identical_images(self, reference_image):
        value = similarity.MSE().compare(reference_image, reference_image)
        assert value == 0.0


class TestPSNR:
    def test_compare_returns_correct_result(self, reference_image, target_image):
        value = similarity.PSNR().compare(target_image, reference_image)
        assert round(value, 2) == 46.88

# class TestSSIM:
#     def test_comapare_returns_one_when_images_are_identical(self):
#         reference_image = np.ones((10, 10))
#         value = similarity.SSIM().compare(reference_image, reference_image)
#         assert value == 1.0
