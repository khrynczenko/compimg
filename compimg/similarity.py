"""Module with routines for computing similarity between images"""
import abc
import numpy as np
import compimg

from compimg import kernels
from compimg._internals import _decorators, _utilities

# Kernel that is used in the SSIM implementation presented by the authors.
_SSIM_GAUSSIAN_KERNEL_11X11 = np.array([
    [1.0576e-06, 7.8144e-06, 3.7022e-05, 0.00011246, 0.00021905, 0.00027356, 0.00021905, 0.00011246, 3.7022e-05,
     7.8144e-06, 1.0576e-06],
    [7.8144e-06, 5.7741e-05, 0.00027356, 0.00083101, 0.0016186, 0.0020214, 0.0016186, 0.00083101, 0.00027356,
     5.7741e-05, 7.8144e-06],
    [3.7022e-05, 0.00027356, 0.0012961, 0.0039371, 0.0076684, 0.0095766, 0.0076684, 0.0039371, 0.0012961,
     0.00027356, 3.7022e-05],
    [0.00011246, 0.00083101, 0.0039371, 0.01196, 0.023294, 0.029091, 0.023294, 0.01196, 0.0039371, 0.00083101,
     0.00011246],
    [0.00021905, 0.0016186, 0.0076684, 0.023294, 0.045371, 0.056662, 0.045371, 0.023294, 0.0076684, 0.0016186,
     0.00021905],
    [0.00027356, 0.0020214, 0.0095766, 0.029091, 0.056662, 0.070762, 0.056662, 0.029091, 0.0095766, 0.0020214,
     0.00027356],
    [0.00021905, 0.0016186, 0.0076684, 0.023294, 0.045371, 0.056662, 0.045371, 0.023294, 0.0076684, 0.0016186,
     0.00021905],
    [0.00011246, 0.00083101, 0.0039371, 0.01196, 0.023294, 0.029091, 0.023294, 0.01196, 0.0039371, 0.00083101,
     0.00011246],
    [3.7022e-05, 0.00027356, 0.0012961, 0.0039371, 0.0076684, 0.0095766, 0.0076684, 0.0039371, 0.0012961,
     0.00027356, 3.7022e-05],
    [7.8144e-06, 5.7741e-05, 0.00027356, 0.00083101, 0.0016186, 0.0020214, 0.0016186, 0.00083101, 0.00027356,
     5.7741e-05, 7.8144e-06],
    [1.0576e-06, 7.8144e-06, 3.7022e-05, 0.00011246, 0.00021905, 0.00027356, 0.00021905, 0.00011246, 3.7022e-05,
     7.8144e-06, 1.0576e-06]])


class SimilarityMetric(abc.ABC):
    """
    Abstract class for all similarity metrics.

    """

    @abc.abstractmethod
    def compare(self, image: np.ndarray, reference: np.ndarray) -> float:
        """
        Performs comparison.

        :param image: Image that is being compared.
        :param reference: Image that we compare to.
        :return: Numerical result of the comparison.
        """


class MSE(SimilarityMetric):
    """
    Mean squared error.

    """

    @_decorators._raise_when_arrays_have_different_dtypes
    @_decorators._raise_when_arrays_have_different_shapes
    def compare(self, image: np.ndarray, reference: np.ndarray) -> float:
        image = image.astype(compimg.intermediate_type, copy=False)
        reference = reference.astype(compimg.intermediate_type, copy=False)
        return np.mean(((reference - image) ** 2))


class RMSE(SimilarityMetric):
    """
    Root mean squared error.

    """

    @_decorators._raise_when_arrays_have_different_dtypes
    @_decorators._raise_when_arrays_have_different_shapes
    def compare(self, image: np.ndarray, reference: np.ndarray) -> float:
        image = image.astype(compimg.intermediate_type, copy=False)
        reference = reference.astype(compimg.intermediate_type, copy=False)
        return np.sqrt(MSE().compare(image, reference))


class MAE(SimilarityMetric):
    """
    Mean absolute error.

    """

    @_decorators._raise_when_arrays_have_different_dtypes
    @_decorators._raise_when_arrays_have_different_shapes
    def compare(self, image: np.ndarray, reference: np.ndarray) -> float:
        image = image.astype(compimg.intermediate_type, copy=False)
        reference = reference.astype(compimg.intermediate_type, copy=False)
        return np.mean(np.abs(reference - image))


class PSNR(SimilarityMetric):
    """
    Peak signal-to-noise ratio according to
    https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio.

    """

    @_decorators._raise_when_arrays_have_different_dtypes
    @_decorators._raise_when_arrays_have_different_shapes
    def compare(self, image: np.ndarray, reference: np.ndarray) -> float:
        image_original_dtype = image.dtype
        image = image.astype(compimg.intermediate_type, copy=False)
        reference = reference.astype(compimg.intermediate_type, copy=False)
        mse = MSE().compare(image, reference)
        if mse == 0.0:
            return float("inf")
        _, max_pixel_value = _utilities._get_image_dtype_range(
            image_original_dtype)
        psnr = 20 * np.log10(max_pixel_value) - 10 * np.log10(mse)
        return psnr


class SSIM(SimilarityMetric):
    """
    Structural similarity index according to the paper from 2004
    "Image Quality Assessment: From Error Visibility to Structural Similarity"
    by Wang et al.
    In case you would like to change alpha, beta and gamma parameters you
    could change private attributes _alpha. _beta and _gamma respectively.

    """

    def __init__(self, k1: float = 0.01, k2: float = 0.03):
        self._k1 = k1
        self._k2 = k2

    @_decorators._raise_when_arrays_have_different_dtypes
    @_decorators._raise_when_arrays_have_different_shapes
    def compare(self, image: np.ndarray, reference: np.ndarray) -> float:
        _, max_pixel_value = _utilities._get_image_dtype_range(image.dtype)
        C1 = (self._k1 * max_pixel_value) ** 2
        C2 = (self._k2 * max_pixel_value) ** 2
        image = image.astype(compimg.intermediate_type)
        reference = reference.astype(compimg.intermediate_type)
        I1 = image
        I2 = reference
        I2_2 = reference * reference
        I1_2 = image * image
        I1_I2 = image * reference
        mu1 = kernels._convolve_without_clipping_changing_dtype(I1, _SSIM_GAUSSIAN_KERNEL_11X11)
        mu2 = kernels._convolve_without_clipping_changing_dtype(I2, _SSIM_GAUSSIAN_KERNEL_11X11)
        mu1_2 = mu1 * mu1
        mu2_2 = mu2 * mu2
        mu1_mu2 = mu1 * mu2
        sigma1_2 = kernels._convolve_without_clipping_changing_dtype(I1_2, _SSIM_GAUSSIAN_KERNEL_11X11)
        sigma1_2 -= mu1_2
        sigma2_2 = kernels._convolve_without_clipping_changing_dtype(I2_2, _SSIM_GAUSSIAN_KERNEL_11X11)
        sigma2_2 -= mu2_2
        sigma12 = kernels._convolve_without_clipping_changing_dtype(I1_I2, _SSIM_GAUSSIAN_KERNEL_11X11)
        sigma12 -= mu1_mu2

        t1 = 2 * mu1_mu2 + C1
        t2 = 2 * sigma12 + C2
        t3 = t1 * t2

        t1 = mu1_2 + mu2_2 + C1
        t2 = sigma1_2 + sigma2_2 + C2
        t1 = t1 * t2
        ssim_map = t3 / t1
        return np.mean(ssim_map)


class GSSIM():
    """
    Gradient-Based Structural similarity index according to the paper
    "GRADIENT-BASED STRUCTURAL SIMILARITY FOR IMAGE QUALITY ASSESSMENT"
    by Chen et al.
    In case you would like to change alpha, beta and gamma parameters you
    could change private attributes _alpha. _beta and _gamma respectively.

    """
