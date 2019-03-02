"""
Here is the simple example of how one can compare one image to another.

>>> import numpy as np
>>> from compimg.similarity import MSE
>>> img = np.ones((10,10), dtype = np.uint8)
>>> error = MSE().compare(img, img)
>>> # comparison of two identical images using MSE so output should be 0
>>> assert error == 0.0

"""
