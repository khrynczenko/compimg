import functools
import inspect
import numpy as np

from collections import OrderedDict
from inspect import BoundArguments


def raise_when_arrays_have_different_shapes(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        signature = inspect.signature(func)
        bound_arguments: BoundArguments = signature.bind(*args, **kwargs)
        all_args: OrderedDict = bound_arguments.arguments
        image: np.ndarray = all_args.get("image")
        reference: np.ndarray = all_args.get("reference")
        if image.shape != reference.shape:
            raise ValueError("Both images must be of the same shape.")
        return func(*bound_arguments.args, **bound_arguments.kwargs)

    return wrapper


def raise_when_arrays_have_different_dtypes(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        signature = inspect.signature(func)
        bound_arguments: BoundArguments = signature.bind(*args, **kwargs)
        all_args: OrderedDict = bound_arguments.arguments
        image: np.ndarray = all_args.get("image")
        reference: np.ndarray = all_args.get("reference")
        if image.dtype != reference.dtype:
            raise ValueError("Both images must be of the same dtype.")
        return func(*bound_arguments.args, **bound_arguments.kwargs)

    return wrapper
