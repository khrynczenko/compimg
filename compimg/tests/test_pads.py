import numpy as np
import pytest

from typing import List
from numbers import Number
from compimg.exceptions import NegativePadAmountError
from compimg.pads import FromFunctionPad, ConstantPad, EdgePad


def get_borders_as_1d_list(image: np.ndarray) -> List[Number]:
    return list(image[0, :]) + list(image[-1, :]) + list(
        image[1:-1, 0]) + list(image[1:-1, -1])


class TestFromFunctionPad:
    def test_if_pads_correctly(self):
        pad = FromFunctionPad(lambda x: x)
        image = np.ones((3, 3))
        assert np.array_equal(pad.apply(image), image)


class TestConstantPad:

    @pytest.mark.parametrize("value, amount, image_shape", [
        (0, 1, (3, 3)),
        (0, 2, (3, 3)),
        (0, 0, (3, 3)),
        (1, 1, (3, 3)),
        (0, 1, (3, 3, 3)),
        (0, 2, (3, 3, 3)),
        (0, 0, (3, 3, 3)),
        (1, 1, (3, 3, 3)),
    ])
    def test_if_pads_correctly_with_usual_parameters(self, value, amount,
                                                     image_shape):
        pad = ConstantPad(value, amount)
        image = np.ones(image_shape)
        padded_image = pad.apply(image)
        channels = 3 if image.ndim == 3 else 1
        expected_sum = (image.size * 1) + sum(
            [value * (2 ** (3 + amount)) * channels for _ in range(amount)])
        assert expected_sum == np.sum(padded_image)

    def test_if_raises_when_initialized_with_negative_amount(self):
        with pytest.raises(NegativePadAmountError):
            ConstantPad(0, -1)


class TestEdgePad:

    @pytest.mark.parametrize("amount", [0, 1])
    def test_if_pads_correctly_with_usual_parameters(self, amount):
        pad = EdgePad(amount)
        image = np.ones((3, 3))
        padded_image = pad.apply(image)
        assert np.sum(padded_image) - np.sum(image) == 1 * (
                padded_image.size - image.size)

    def test_if_raises_when_initialized_with_negative_amount(self):
        with pytest.raises(NegativePadAmountError):
            EdgePad(-1)
