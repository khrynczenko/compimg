"""
Here is the simple example of how one can compare one image to another.

>>> import numpy as np
>>> from compimg.similarity import MSE
>>> img = np.ones((20,20), dtype = np.uint8)
>>> reference = np.ones((20,20), dtype = np.uint8)
>>> MSE().compare(img, img)
0.0

All metrics implement single interface so it is easy to use multiple of them
for example you could run:

>>> import numpy as np
>>> from compimg.similarity import MSE, PSNR, SSIM
>>> for metric in [MSE(), PSNR(), SSIM()]:
...     img = np.ones((20,20), dtype = np.uint8)
...     reference = np.zeros((20,20), dtype = np.uint8)
...     value = round(metric.compare(img, reference), 2)
...     print(f"{metric.__class__.__name__} = {value}")
MSE = 1.0
PSNR = 48.13
SSIM = 0.87

compimg implicitly converts image to intermediate type (float64) to avoid
overflow/underflow when doing calculation. Its advised to leave this type as
is, albeit it is possible to change it. For example you could sacrafice
precision to improve processing speed by changing it to float32 or even
float16.

>>> import numpy as np
>>> import compimg
>>> import compimg.similarity
>>> compimg.config.intermediate_type = np.dtype(np.float32)
>>> # code that uses similarity metrics


"""
import numpy as np


class Config:
    def __init__(self):
        # This is type that is used for all the calculations (images are
        # converted into it if necessary, for example when overflow or
        # underflow would occur due to calculations).
        # Change only if you know what you are doing.
        self.intermediate_dtype: np.dtype = np.dtype(np.float64)


config = Config()
