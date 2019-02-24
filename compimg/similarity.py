"""Module with routines for computing similarity between images"""
import abc
import numpy as np

from compimg._internals import _decorators, _utilities


class SimilarityMetric(abc.ABC):
    """Abstract class for all similarity metrics."""

    @abc.abstractmethod
    def compare(self, image: np.ndarray, reference: np.ndarray) -> float:
        """
        Performs comparison.
        :param image: Image that is being compared.
        :param reference: Image that we compare to.
        :return: Numerical result of the comparison.
        """
        pass


class MSE(SimilarityMetric):
    """Mean squared error"""

    @_decorators.are_arrays_of_the_same_dtype
    @_decorators.are_arrays_of_the_same_shape
    def compare(self, image: np.ndarray, reference: np.ndarray) -> float:
        return np.sum(((reference - image) ** 2)) / image.size


class PSNR(SimilarityMetric):
    """Peak signal-to-noise ratio according to
        https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
    """

    @_decorators.are_arrays_of_the_same_dtype
    @_decorators.are_arrays_of_the_same_shape
    def compare(self, image: np.ndarray, reference: np.ndarray) -> float:
        mse = MSE().compare(image, reference)
        if mse == 0.0:
            return float("inf")
        _, max_pixel_value = _utilities.get_dtype_range(image.dtype)
        psnr = 20 * np.log10(max_pixel_value) - 10 * np.log10(mse)
        return psnr


class SSIM(SimilarityMetric):
    """
    Structural similarity index according to
        https://ipfs.io/ipfs/QmXoypizjW3WknFiJnKLwHCnL72vedxjQkDDP1mXWo6uco/wiki/Structural_similarity.html
    """

    def __init__(self, k1: float = 0.01, k2: float = 0.03):
        self.k1 = k1
        self.k2 = k2

    @_decorators.are_arrays_of_the_same_dtype
    @_decorators.are_arrays_of_the_same_shape
    def compare(self, image: np.ndarray, reference: np.ndarray) -> float:
        x_avg = image.mean()
        y_avg = reference.mean()
        x_var = image.var()
        y_var = reference.var()
        x_y_cov = x_avg * y_avg - x_avg * y_avg
        _, maximum_pixel_value = _utilities.get_dtype_range(image.dtype)
        c1 = (self.k1 * maximum_pixel_value) ** 2
        c2 = (self.k2 * maximum_pixel_value) ** 2
        nominator = (2.0 * x_avg * y_avg + c1) * (2.0 * x_y_cov + c2)
        denominator = (x_avg ** 2 + y_avg ** 2 + c1) * (x_var ** 2 + y_var ** 2 + c2)
        ssim = nominator / denominator
        return ssim
