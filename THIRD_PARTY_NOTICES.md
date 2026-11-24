# Third-party model notices

## MobileNet-SSD

The object detector files (`MobileNetSSD_deploy.prototxt` and `mobilenet_iter_73000.caffemodel`) are fetched from the pinned revision `bb17b6c3eef36d80be441ae8e5339be66e8e3b7a` of [chuanqi305/MobileNet-SSD](https://github.com/chuanqi305/MobileNet-SSD) during Docker image builds. The upstream repository provides this notice:

> MIT License
>
> Copyright (c) 2018 chuanqi305
>
> Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
>
> The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

The OpenCV face-presence cascade (`haarcascade_frontalface_default.xml`) is distributed with the pinned `opencv-python-headless` package. Its upstream source and license header are available in the [OpenCV repository](https://github.com/opencv/opencv/blob/4.x/data/haarcascades/haarcascade_frontalface_default.xml). Face detection reports presence only; it does not identify people.
