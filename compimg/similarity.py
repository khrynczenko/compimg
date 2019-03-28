"""Module with routines for computing similarity between images"""
import abc
import numpy as np

from numbers import Real
from compimg import kernels
from compimg._internals import _decorators, _utilities
from compimg.pads import EdgePad
from compimg.windows import SlidingWindow, IdentitySlidingWindow, \
    GaussianSlidingWindow

# This is type that is used for all the calculations (images are
# converted into it if necessary, for example when overflow or underflow
# would occur due to calculations).
# Change only if you know what you are doing.
intermediate_type: np.dtype = np.float64


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
        image = image.astype(intermediate_type, copy=False)
        reference = reference.astype(intermediate_type, copy=False)
        return np.mean(((reference - image) ** 2))


class RMSE(SimilarityMetric):
    """
    Root mean squared error.

    """

    @_decorators._raise_when_arrays_have_different_dtypes
    @_decorators._raise_when_arrays_have_different_shapes
    def compare(self, image: np.ndarray, reference: np.ndarray) -> float:
        image = image.astype(intermediate_type, copy=False)
        reference = reference.astype(intermediate_type, copy=False)
        return np.sqrt(MSE().compare(image, reference))


class MAE(SimilarityMetric):
    """
    Mean absolute error.

    """

    @_decorators._raise_when_arrays_have_different_dtypes
    @_decorators._raise_when_arrays_have_different_shapes
    def compare(self, image: np.ndarray, reference: np.ndarray) -> float:
        image = image.astype(intermediate_type, copy=False)
        reference = reference.astype(intermediate_type, copy=False)
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
        image = image.astype(intermediate_type, copy=False)
        reference = reference.astype(intermediate_type, copy=False)
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

    def __init__(self, k1: float = 0.01, k2: float = 0.03,
                 sliding_window: SlidingWindow = GaussianSlidingWindow(
                     shape=(11, 11), stride=(1, 1), sigma=1.5)):
        self._k1 = k1
        self._k2 = k2
        self._alpha = 1
        self._beta = 1
        self._gamma = 1
        self._sliding_window = sliding_window

    @_decorators._raise_when_arrays_have_different_dtypes
    @_decorators._raise_when_arrays_have_different_shapes
    def compare(self, image: np.ndarray, reference: np.ndarray) -> float:
        _, max_pixel_value = _utilities._get_image_dtype_range(image.dtype)
        image = image.astype(intermediate_type, copy=False)
        reference = reference.astype(intermediate_type, copy=False)
        image_windows = self._sliding_window.slide(image)
        reference_windows = self._sliding_window.slide(reference)
        windows_results = []
        for window, reference_window in zip(image_windows, reference_windows):
            windows_results.append(self._calculate_on_window(window,
                                                             reference_window,
                                                             max_pixel_value))
        return np.mean(windows_results)

    def _calculate_on_window(self, image: np.ndarray,
                             reference: np.ndarray,
                             max_pixel_value: Real) -> float:
        x_avg = image.mean()
        y_avg = reference.mean()
        x_var = np.sum((image - x_avg) ** 2) / (image.size - 1)
        y_var = np.sum((reference - y_avg) ** 2) / (reference.size - 1)
        x_var_square_root = np.sqrt(x_var)
        y_var_square_root = np.sqrt(y_var)
        x_y_cov = (np.sum(image - x_avg) * np.sum(reference - y_avg)
                   / (image.size - 1))
        c1 = (self._k1 * float(max_pixel_value)) ** 2
        c2 = (self._k2 * float(max_pixel_value)) ** 2
        c3 = c2 / 2
        luminance = (2 * x_avg * y_avg + c1) / (x_avg ** 2 + y_avg ** 2 + c1)
        contrast = (2 * x_var_square_root * y_var_square_root + c2) / (
                x_var + y_var + c2)
        structure = (x_y_cov + c3) / (
                x_var_square_root * y_var_square_root + c3)
        return luminance ** self._alpha * contrast ** self._beta * (
                structure ** self._gamma)


class GSSIM(SimilarityMetric):
    """
    Gradien-Based Structural similarity index according to the paper
    "GRADIENT-BASED STRUCTURAL SIMILARITY FOR IMAGE QUALITY ASSESSMENT"
    by Chen et al.
    In case you would like to change alpha, beta and gamma parameters you
    could change private attributes _alpha. _beta and _gamma respectively.

    """

    def __init__(self, k1: float = 0.01, k2: float = 0.03,
                 sliding_window: SlidingWindow = GaussianSlidingWindow(
                     shape=(11, 11), stride=(1, 1), sigma=1.5)):
        self._k1 = k1
        self._k2 = k2
        self._alpha = 1
        self._beta = 1
        self._gamma = 1
        self._sliding_window = sliding_window

    @_decorators._raise_when_arrays_have_different_dtypes
    @_decorators._raise_when_arrays_have_different_shapes
    def compare(self, image: np.ndarray, reference: np.ndarray) -> float:
        _, max_pixel_value = _utilities._get_image_dtype_range(image.dtype)
        image = image.astype(intermediate_type, copy=False)
        reference = reference.astype(intermediate_type, copy=False)
        padded_image = EdgePad(2).apply(image)
        x_image = kernels.convolve(padded_image, kernels.VERTICAL_SOBEL_3x3)
        y_image = kernels.convolve(padded_image, kernels.HORIZONTAL_SOBEL_3x3)
        gradient_map = np.sqrt(x_image ** 2 + y_image ** 2)
        image_windows = self._sliding_window.slide(image)
        gradient_windows = self._sliding_window.slide(gradient_map)
        reference_windows = self._sliding_window.slide(reference)
        windows_results = []
        for window, gradient_window, reference_window in zip(
                image_windows,
                gradient_windows,
                reference_windows):
            windows_results.append(self._calculate_on_window(window,
                                                             gradient_window,
                                                             reference_window,
                                                             max_pixel_value))
        return np.mean(windows_results)

    def _calculate_on_window(self, image: np.ndarray,
                             gradient_map: np.ndarray,
                             reference: np.ndarray,
                             max_pixel_value: Real) -> float:
        x_avg = image.mean()
        y_avg = reference.mean()
        gradient_map_avg = gradient_map.mean()
        gradient_map_var = np.sum((gradient_map - gradient_map_avg) ** 2) / (
                gradient_map.size - 1)
        y_var = np.sum((reference - y_avg) ** 2) / (reference.size - 1)
        gradient_map_var_square_root = np.sqrt(gradient_map_var)
        y_var_square_root = np.sqrt(y_var)
        gradient_map_y_cov = (np.sum(gradient_map - gradient_map_avg) * np.sum(
            reference - y_avg)
                              / (gradient_map.size - 1))
        c1 = (self._k1 * float(max_pixel_value)) ** 2
        c2 = (self._k2 * float(max_pixel_value)) ** 2
        c3 = c2 / 2
        luminance = (2 * x_avg * y_avg + c1) / (x_avg ** 2 + y_avg ** 2 + c1)
        contrast = (2 * gradient_map_var_square_root * y_var_square_root +
                    c2) / (gradient_map_var + y_var + c2)
        structure = (gradient_map_y_cov + c3) / (
                gradient_map_var_square_root * y_var_square_root + c3)
        return luminance ** self._alpha * contrast ** self._beta * (
                structure ** self._gamma)
