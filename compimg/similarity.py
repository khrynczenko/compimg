"""Module with routines for computing similarity between images"""
import enum
import numpy as np


class Metric(enum.IntEnum):
    PSNR = 1


def compute_similarity(metric: Metric, reference: np.ndarray, target: np.ndarray) -> float:
    """
    Computes given metric for how target compares to reference.
    :param metric: Which metric to use.
    :param reference: Image to which we compare to.
    :param target: Image that is being compared.
    :return: Similarity value.
    """
    pass
