# CHANGELOG

## compimg 0.2.2

- Fixed documentation for `similarity` module (docs for metrics would
  not appear)
- Python 3.8 officially supported (checking added to CI)
- Improve codebase by introducing black for formatting
- Added simple benchmarking so differences can be measured
  when changes to existing code are made

## compimg 0.2.1

- Improved performance of SSIM and GSSIM.
- Now using scipy to perform convolutions. Due to that now  `compimg` is
dependent on `scipy`.
- Fixed issue where `_internals` package could not be found.

## compimg 0.2.0

- Added `GSSIM` metric
- Added `RMSE` metric
- Added `MAE' metric
- Added `compimg.pads` module which provides easy way to apply padding to an
image (used in *SSIM implementations)
- Added`compimg.kernels` module which makes possible applying kernel to an
image (used within *SSIM implementations)
- More and better exceptions
- Moved `compimg.similarity.intermediate_type` to
`compimg.config.intermediate_dtype`
- Fixed `SSIM` metric (now implementation follows steps from the one
provided by authors)

## compimg 0.1.1

This release fixes some small documentation errors, readme typos and
adds some badges to the README file. There are no actual code changes.

