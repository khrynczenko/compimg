import inspect
import functools
import collections
import numpy as np


def are_arrays_of_the_same_shape(func):
    functools.wraps(func)

    def wrapper(*args, **kwargs):
        signature = inspect.signature(func)
        bound_arguments: inspect.BoundArguments = signature.bind(*args, **kwargs)
        all_args: collections.OrderedDict = bound_arguments.arguments
        image: np.ndarray = all_args.get("image")
        reference: np.ndarray = all_args.get("reference")
        if image.shape != reference.shape:
            raise ValueError("Both images must be of the same shape.")
        return func(*bound_arguments.args, **bound_arguments.kwargs)

    functools.update_wrapper(wrapper, func)

    return wrapper


def are_arrays_of_the_same_dtype(func):
    functools.wraps(func)

    def wrapper(*args, **kwargs):
        signature = inspect.signature(func)
        bound_arguments: inspect.BoundArguments = signature.bind(*args, **kwargs)
        all_args: collections.OrderedDict = bound_arguments.arguments
        image: np.ndarray = all_args.get("image")
        reference: np.ndarray = all_args.get("reference")
        if image.dtype != reference.dtype:
            raise ValueError("Both images must be of the same dtype.")
        return func(*bound_arguments.args, **bound_arguments.kwargs)

    functools.update_wrapper(wrapper, func)

    return wrapper
