# CHANGELOG

## compimg 0.2.0
- Added `GSSIM` metric
- Added `RMSE` metric
- Added `MAE' metric
- Added `compimg.pads` module which provides easy way to apply padding to an image (used in 
*SSIM implementations)
- Added `compimg.kernels` module which makes possible applying kernel to an image (used 
within *SSIM implementations)
- More and better exceptions
- Moved `compimg.similarity.intermediate_type` to `compimg.config.intermediate_dtype`
- Fixed `SSIM` metric (now implementation follows steps from the one provided by authors)

## compimg 0.1.1
This release fixes some small documentation errors, readme typos and 
adds some badges to the README file. There are no actual code changes.

