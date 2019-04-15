"""Module with routines for computing similarity between images"""
import abc
import numpy as np
import compimg

from compimg import kernels
from compimg.pads import EdgePad
from compimg._internals import _decorators, _utilities

# Kernel that is used in the SSIM implementation presented by the authors in
# "Image Quality Assessment: From Error Visibility to Structural Similarity"
# by Wang et al.
_SSIM_GAUSSIAN_KERNEL_11X11 = np.array([
    [1.0576e-06, 7.8144e-06, 3.7022e-05, 0.00011246, 0.00021905, 0.00027356,
     0.00021905, 0.00011246, 3.7022e-05, 7.8144e-06, 1.0576e-06],
    [7.8144e-06, 5.7741e-05, 0.00027356, 0.00083101, 0.0016186, 0.0020214,
     0.0016186, 0.00083101, 0.00027356, 5.7741e-05, 7.8144e-06],
    [3.7022e-05, 0.00027356, 0.0012961, 0.0039371, 0.0076684, 0.0095766,
     0.0076684, 0.0039371, 0.0012961, 0.00027356, 3.7022e-05],
    [0.00011246, 0.00083101, 0.0039371, 0.01196, 0.023294, 0.029091, 0.023294,
     0.01196, 0.0039371, 0.00083101, 0.00011246],
    [0.00021905, 0.0016186, 0.0076684, 0.023294, 0.045371, 0.056662, 0.045371,
     0.023294, 0.0076684, 0.0016186, 0.00021905],
    [0.00027356, 0.0020214, 0.0095766, 0.029091, 0.056662, 0.070762, 0.056662,
     0.029091, 0.0095766, 0.0020214, 0.00027356],
    [0.00021905, 0.0016186, 0.0076684, 0.023294, 0.045371, 0.056662, 0.045371,
     0.023294, 0.0076684, 0.0016186, 0.00021905],
    [0.00011246, 0.00083101, 0.0039371, 0.01196, 0.023294, 0.029091, 0.023294,
     0.01196, 0.0039371, 0.00083101, 0.00011246],
    [3.7022e-05, 0.00027356, 0.0012961, 0.0039371, 0.0076684, 0.0095766,
     0.0076684, 0.0039371, 0.0012961, 0.00027356, 3.7022e-05],
    [7.8144e-06, 5.7741e-05, 0.00027356, 0.00083101, 0.0016186, 0.0020214,
     0.0016186, 0.00083101, 0.00027356, 5.7741e-05, 7.8144e-06],
    [1.0576e-06, 7.8144e-06, 3.7022e-05, 0.00011246, 0.00021905, 0.00027356,
     0.00021905, 0.00011246, 3.7022e-05, 7.8144e-06, 1.0576e-06]])


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
        image = image.astype(compimg.config.intermediate_dtype, copy=False)
        reference = reference.astype(compimg.config.intermediate_dtype,
                                     copy=False)
        return np.mean(((reference - image) ** 2))


class RMSE(SimilarityMetric):
    """
    Root mean squared error.

    """

    @_decorators._raise_when_arrays_have_different_dtypes
    @_decorators._raise_when_arrays_have_different_shapes
    def compare(self, image: np.ndarray, reference: np.ndarray) -> float:
        image = image.astype(compimg.config.intermediate_dtype, copy=False)
        reference = reference.astype(compimg.config.intermediate_dtype,
                                     copy=False)
        return np.sqrt(MSE().compare(image, reference))


class MAE(SimilarityMetric):
    """
    Mean absolute error.

    """

    @_decorators._raise_when_arrays_have_different_dtypes
    @_decorators._raise_when_arrays_have_different_shapes
    def compare(self, image: np.ndarray, reference: np.ndarray) -> float:
        image = image.astype(compimg.config.intermediate_dtype, copy=False)
        reference = reference.astype(compimg.config.intermediate_dtype,
                                     copy=False)
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
        image = image.astype(compimg.config.intermediate_dtype, copy=False)
        reference = reference.astype(compimg.config.intermediate_dtype,
                                     copy=False)
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

    """

    def __init__(self, k1: float = 0.01, k2: float = 0.03):
        self._k1 = k1
        self._k2 = k2

    @_decorators._raise_when_arrays_have_different_dtypes
    @_decorators._raise_when_arrays_have_different_shapes
    def compare(self, image: np.ndarray, reference: np.ndarray) -> float:
        # This implementations is based on
        # https://docs.opencv.org/2.4/doc/tutorials/highgui/video-input-psnr-ssim/video-input-psnr-ssim.html
        convolve = kernels._unobtrusive_convolve
        _, max_pixel_value = _utilities._get_image_dtype_range(image.dtype)
        C1 = (self._k1 * max_pixel_value) ** 2
        C2 = (self._k2 * max_pixel_value) ** 2
        image = image.astype(compimg.config.intermediate_dtype)
        reference = reference.astype(compimg.config.intermediate_dtype)
        x = image
        y = reference
        y_squared = reference * reference
        x_squared = image * image
        x_times_y = image * reference
        x_mean = convolve(x, _SSIM_GAUSSIAN_KERNEL_11X11)
        y_mean = convolve(y, _SSIM_GAUSSIAN_KERNEL_11X11)
        x_mean_squared = x_mean * x_mean
        y_mean_squared = y_mean * y_mean
        sigma_x_squared = convolve(x_squared, _SSIM_GAUSSIAN_KERNEL_11X11)
        sigma_x_squared -= x_mean_squared
        sigma_y_squared = convolve(y_squared, _SSIM_GAUSSIAN_KERNEL_11X11)
        sigma_y_squared -= y_mean_squared
        sigma_x_y = convolve(x_times_y, _SSIM_GAUSSIAN_KERNEL_11X11)
        sigma_x_y -= x_mean * y_mean

        t1 = 2 * x_mean * y_mean + C1
        t2 = 2 * sigma_x_y + C2
        t3 = t1 * t2

        t1 = x_mean_squared + y_mean_squared + C1
        t2 = sigma_x_squared + sigma_y_squared + C2
        t1 = t1 * t2
        ssim_map = t3 / t1
        return np.mean(ssim_map)


class GSSIM(SimilarityMetric):
    """
    Gradient-Based Structural similarity index according to the paper
    "GRADIENT-BASED STRUCTURAL SIMILARITY FOR IMAGE QUALITY ASSESSMENT"
    by Chen et al.

    """

    def __init__(self, k1: float = 0.01, k2: float = 0.03):
        self._k1 = k1
        self._k2 = k2

    @_decorators._raise_when_arrays_have_different_dtypes
    @_decorators._raise_when_arrays_have_different_shapes
    def compare(self, image: np.ndarray, reference: np.ndarray) -> float:
        convolve = kernels._unobtrusive_convolve
        _, max_pixel_value = _utilities._get_image_dtype_range(image.dtype)
        C1 = (self._k1 * max_pixel_value) ** 2
        C2 = (self._k2 * max_pixel_value) ** 2
        C3 = C2 / 2.0
        image = image.astype(compimg.config.intermediate_dtype)
        reference = reference.astype(compimg.config.intermediate_dtype)
        x = image
        y = reference
        x_mean = convolve(x, _SSIM_GAUSSIAN_KERNEL_11X11)
        y_mean = convolve(y, _SSIM_GAUSSIAN_KERNEL_11X11)
        x_mean_squared = x_mean * x_mean
        y_mean_squared = y_mean * y_mean

        sobel_image = self._apply_sobel(image)
        sobel_reference = self._apply_sobel(reference)
        # sx means sobel_x
        sx_squared = sobel_image * sobel_image
        sy_squared = sobel_reference * sobel_reference
        sxy = sobel_reference * sobel_image
        sx_mean = convolve(sobel_image, _SSIM_GAUSSIAN_KERNEL_11X11)
        sy_mean = convolve(sobel_reference, _SSIM_GAUSSIAN_KERNEL_11X11)
        sx_mean_squared = sx_mean * sx_mean
        sy_mean_squared = sy_mean * sy_mean

        sigma_sx_squared = convolve(sx_squared, _SSIM_GAUSSIAN_KERNEL_11X11)
        sigma_sx_squared -= sx_mean_squared
        sigma_sy_squared = convolve(sy_squared, _SSIM_GAUSSIAN_KERNEL_11X11)
        sigma_sy_squared -= sy_mean_squared
        sigma_sxy = convolve(sxy, _SSIM_GAUSSIAN_KERNEL_11X11)
        sigma_sxy -= sx_mean * sy_mean
        luminance = (2 * x_mean * y_mean + C1) / (
                x_mean_squared + y_mean_squared + C1)
        contrast = (2 * np.sqrt(sigma_sx_squared) * np.sqrt(
            sigma_sy_squared) + C2) / (
                           sigma_sx_squared + sigma_sy_squared + C2)
        structure = (sigma_sxy + C3) / (
                np.sqrt(sigma_sx_squared) * np.sqrt(sigma_sy_squared) + C3)
        return np.mean(luminance * contrast * structure)

    def _apply_sobel(self, array: np.ndarray) -> np.ndarray:
        convolve = kernels._unobtrusive_convolve
        array = EdgePad(1).apply(array)
        array1 = convolve(array, kernels.HORIZONTAL_SOBEL_3x3)
        array2 = convolve(array, kernels.VERTICAL_SOBEL_3x3)
        return array1 + array2
