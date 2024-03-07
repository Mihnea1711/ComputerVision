# ComputerVision
A repository featuring labs from my university's Computer Vision course, demonstrating practical applications in image processing, object detection, and segmentation.

## Computer Vision Course Projects

In this repository, I'll be documenting my progress through the Computer Vision course at my university. Throughout the course, we'll progressively implement five projects:

Project 1: Lane Detection
Project 2: Digit Recognition
Project 3: Image Stitching - Creating panoramas
Project 4: Stereo
Project 5: Object Detection using neural networks

All projects will be implemented using Python programming language and Jupyter notebooks.

Stay tuned for updates and explorations into the fascinating world of computer vision!

### Week #1 Lane Detection Application
This project implements a lane detection application using Python, Jupyter Notebook, NumPy, and OpenCV. The application detects lane markings in images and overlays them with visualizations to highlight the detected lanes.

#### Overview
The lane detection pipeline consists of the following steps:
- Read the test image.
- Convert the image from BGR to RGB.
- Apply grayscale transformation.
- Apply Gaussian smoothing.
- Perform Canny edge detection.
- Define a region of interest.
- Apply a mask to the Canny edges.
- Overlay the detected edges on the original image.

### Week #2 Advanced Lane Detection by Fitting Lines Using Hough Transform

#### Project Overview
During Week 2 of development, the focus was on utilizing the Hough Transform to fit lines to the detected edges. While Canny edge detection provided edge information, converting these edges into lines required additional processing.

#### Hough Transform Overview:
The Hough Transform is a voting technique used to identify lines in an image.
It converts points in the image space to lines in the Hough space, allowing us to find lines based on edge points.

#### Using Hough Transform for Lane Detection:
Each edge point in the image space contributes to a set of possible parameters (m, b) in the Hough space.
The algorithm records all possible lines on which each edge point lies and identifies lines with the most votes.

#### Key Takeaways:
- Gaussian Blur: Removes noise from the image.
- Canny Edge Detection: Isolates edges in the image.
- Bitwise And Function: Isolates edges corresponding to lane lines.
- Hough Transform: Converts edges into lines for lane detection.