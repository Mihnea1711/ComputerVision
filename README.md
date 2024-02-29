# ComputerVision
A repository featuring labs from my university's Computer Vision course, demonstrating practical applications in image processing, object detection, and segmentation.

## Computer Vision Course Projects

In this repository, I'll be documenting my progress through the Computer Vision course at my university. Throughout the course, we'll progressively implement five projects:

    Project 1: Lane Detection
    Project 2: Digit Recognition
    Project 3: Image Stitching
        Creating panoramas
    Project 4: Stereo
    Project 5: Object Detection using neural networks

All projects will be implemented using Python programming language and Jupyter notebooks.

Stay tuned for updates and explorations into the fascinating world of computer vision!

### Lane Detection Application
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