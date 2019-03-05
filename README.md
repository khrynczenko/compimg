Master: [![CircleCI](https://circleci.com/gh/JenioPY/compimg/tree/master.svg?style=svg&circle-token=08abd49c539289429775861727cae51269c6db2c)](https://circleci.com/gh/JenioPY/compimg/tree/master) 
Develop: [![CircleCI](https://circleci.com/gh/JenioPY/compimg/tree/develop.svg?style=svg&circle-token=08abd49c539289429775861727cae51269c6db2c)](https://circleci.com/gh/JenioPY/compimg/tree/develop)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## Introduction  
Image similarity metrics are often used in image quality assessment for performance
evaluation of image restoration and reconstruction algorithms. They require two images:
- test image (image of interest)
- reference image (image we compare against)  
Such metrics produce numerical value.
 
Such methods are are widely called full/reduced-reference methods for 
assessing image quality.

`compimg` package is all about calculating similarity between images. 
It provides image similarity metrics (PSNR, SSIM etc.) that are widely used 
to asses image quality.

## Features  
- common metrics for calculating similarity of one image to another 
- only `numpy` as a dependency

## Installation
`compimg` is available on PyPI. You can install it using pip:  
`pip install compimg`

## Help
If you have any problems or questions please post an issue.