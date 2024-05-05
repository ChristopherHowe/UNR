import cv2
import PIL
import matplotlib
import skimage
import numpy as np
import math
import sklearn
import project3 as p3

#Extract Vocab
vocab = generate_vocabulary("train_data.txt")

#Train Object Classifier
classifier = train_classifier("train_data.txt", vocab)

#Test Object Classifier
test_img = p3.load_img("test_img.jpg")
out = classify_image(classifier, test_img, vocab)

#Segment an Image
img = p3.load_img("test_img.jpg")
im1, im2, im3 = p3.segment_image(img)
p3.display_img(im1)
p3.display_img(im2)
p3.display_img(im3)

