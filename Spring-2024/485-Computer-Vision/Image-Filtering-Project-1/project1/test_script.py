import cv2
import PIL
import matplotlib
import skimage
import numpy as np
import math
import project1 as p1

filter_size = 5
sigma = 1
theta = math.pi/2

#Load and display image
img = p1.load_img("test_img.jpg")
p1.display_img(img)

#Generate 1D gaussian filter
gaussian1D = generate_gaussian(sigma, filter_size, 1)

#Filter image with 1D gaussian
filtered_img = p1.apply_filter(img, gaussian1D, 0, 0)

#Generate 2D gaussian filter
gaussian2D = generate_gaussian(sigma, filter_size, filter_size)

#Filter image with 2D gaussian
filtered_img = p1.apply_filter(img, gaussian2D, 0, 0)

#Noise removal with median filter
filtered_img = p1.median_filtering(img, filter_size, filter_size)

#Histogram Equalization
filtered_img = p1.hist_eq(img)

#Rotate Image
transformed_img = p1.rotate(img, theta)

#Edge Detection
filtered_img = p1.edge_detection(img)
