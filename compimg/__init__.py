"""
Here is the simple example of how one can compare one image to another.

>>> import numpy as np
>>> from compimg.similarity import MSE
>>> img = np.ones((10,10), dtype = np.uint8)
>>> reference = np.ones((10,10), dtype = np.uint8)
>>> MSE().compare(img, img)
0.0

For multiple metrics you could run:

>>> import numpy as np
>>> from compimg.similarity import MSE, PSNR, SSIM
>>> for metric in [MSE(), PSNR(), SSIM()]:
...     img = np.ones((10,10), dtype = np.uint8)
...     reference = np.zeros((10,10), dtype = np.uint8)
...     value = metric.compare(img, reference)
...     print(f"{metric.__class__.__name__} = {value}")
MSE = 1.0
PSNR = 48.1308036086791
SSIM = 0.8667110963012329

"""
