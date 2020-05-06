import numpy as np
import pytest

from compimg import similarity


@pytest.fixture
def img1():
    return np.ones((1000, 1000), dtype=np.uint8)


@pytest.fixture
def img2():
    return np.zeros((1000, 1000), dtype=np.uint8)


def test_psnr_performance(img1, img2, benchmark):
    psnr = similarity.PSNR()
    benchmark(psnr.compare, img1, img2)


def test_ssim_performance(img1, img2, benchmark):
    ssim = similarity.SSIM()
    benchmark(ssim.compare, img1, img2)


def test_gssim_performance(img1, img2, benchmark):
    gssim = similarity.GSSIM()
    benchmark(gssim.compare, img1, img2)
