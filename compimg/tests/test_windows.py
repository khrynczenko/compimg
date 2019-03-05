import numpy as np

from compimg.windows import DefaultSlidingWindow


class TestDefaultSlidingWindow:
    def test_slide_correctness(self):
        input_image = np.array([[1, 2, 3],
                                [4, 5, 6],
                                [7, 8, 9]])
        slider = DefaultSlidingWindow((2, 2), (1, 1))
        slides = list(slider.slide(input_image))
        assert len(slides) == 4
        assert np.array_equal(slides[0], np.array([[1, 2],
                                                   [4, 5]]))
        assert np.array_equal(slides[1], np.array([[2, 3],
                                                   [5, 6]]))
        assert np.array_equal(slides[2], np.array([[4, 5],
                                                   [7, 8]]))
        assert np.array_equal(slides[3], np.array([[5, 6],
                                                   [8, 9]]))
