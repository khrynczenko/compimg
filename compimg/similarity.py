"""Module with routines for computing similarity between images"""
import abc
import numpy as np

from numbers import Real
from compimg._internals import _decorators, _utilities
from compimg.windows import SlidingWindow, DefaultSlidingWindow


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
        image = image.astype(np.float64, copy=False)
        reference = reference.astype(np.float64, copy=False)
        return np.mean(((reference - image) ** 2))


class RMSE(SimilarityMetric):
    """
    Root mean squared error.

    """

    @_decorators._raise_when_arrays_have_different_dtypes
    @_decorators._raise_when_arrays_have_different_shapes
    def compare(self, image: np.ndarray, reference: np.ndarray) -> float:
        image = image.astype(np.float64, copy=False)
        reference = reference.astype(np.float64, copy=False)
        return np.sqrt(MSE().compare(image, reference))


class MAE(SimilarityMetric):
    """
    Mean absolute error.

    """

    @_decorators._raise_when_arrays_have_different_dtypes
    @_decorators._raise_when_arrays_have_different_shapes
    def compare(self, image: np.ndarray, reference: np.ndarray) -> float:
        image = image.astype(np.float64, copy=False)
        reference = reference.astype(np.float64, copy=False)
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
        image = image.astype(np.float64, copy=False)
        reference = reference.astype(np.float64, copy=False)
        mse = MSE().compare(image, reference)
        if mse == 0.0:
            return float("inf")
        _, max_pixel_value = _utilities.get_dtype_range(image_original_dtype)
        psnr = 20 * np.log10(max_pixel_value) - 10 * np.log10(mse)
        return psnr


class SSIM(SimilarityMetric):
    """
    Structural similarity index according to the paper from 2004
    "Image Quality Assessment: From Error Visibility to Structural Similarity"
    by Wang et al.

    """

    def __init__(self, k1: float = 0.01, k2: float = 0.03,
                 sliding_window: SlidingWindow = DefaultSlidingWindow(
                     size=(8, 8), stride=(1, 1))):
        self._k1 = k1
        self._k2 = k2
        self._sliding_window = sliding_window

    @_decorators._raise_when_arrays_have_different_dtypes
    @_decorators._raise_when_arrays_have_different_shapes
    def compare(self, image: np.ndarray, reference: np.ndarray) -> float:
        _, max_pixel_value = _utilities.get_dtype_range(image.dtype)
        image = image.astype(np.float64, copy=False)
        reference = reference.astype(np.float64, copy=False)
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
        x_y_cov = (np.sum(image - x_avg) * np.sum(reference - y_avg)
                   / (image.size - 1))
        c1 = (self._k1 * float(max_pixel_value)) ** 2
        c2 = (self._k2 * float(max_pixel_value)) ** 2
        nominator = (2.0 * x_avg * y_avg + c1) * (2.0 * x_y_cov + c2)
        denominator = (x_avg ** 2 + y_avg ** 2 + c1) * (
                x_var ** 2 + y_var ** 2 + c2)
        ssim = nominator / denominator
        return ssim
