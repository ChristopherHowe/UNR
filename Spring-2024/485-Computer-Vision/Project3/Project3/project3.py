import cv2
import numpy as np
import math
from typing import List


def load_img(file_name):
    return cv2.imread(file_name)


def display_img(image: np.ndarray):
    cv2.imshow("image filtering project 3", image)
    k = cv2.waitKey(0)
    cv2.destroyAllWindows()


## Remove Helpers ##

def aspect_scale(img: np.ndarray, height):
    ratio = img.shape[0] / img.shape[1]
    return cv2.resize(img, (height, int(height * ratio)))


def display_img_normalized(image: np.ndarray, name: str):
    normImg = np.copy(image)
    newName = name
    # cv2.normalize(image, normImg, 0,255,cv2.NORM_MINMAX)
    if not (np.min(image) == 0 and np.max(image) == 255):
        normImg = np.uint8((normImg - np.min(normImg)) / (np.max(normImg) - np.min(normImg)) * 255)
        newName += ", Normalized"
    if normImg.shape[0] <= 50 or normImg.shape[1] <= 50:
        aspect_scale(normImg, 200)
        newName += ", Scaled"
    cv2.imshow(newName, normImg)
    k = cv2.waitKey(0)
    cv2.destroyAllWindows()

# This function takes a list of training images in txt format. An example is provided with the test script. This function
# should return a set of visual vocabulary “words” in the form of vectors. Each of these vectors will be used as a bin
# in our histogram of features.
def generate_vocabulary(train_data_file: str):
    pass

# This function takes an image and the vocabulary as input and extracts features, generating a BOW count vector.
def extract_features(image, vocabulary):
    pass

# This function takes the training data file and the vocabulary, extracts the features from each training image, and
# trains a classifier (perceptron, KNN, SVM) on the data. You can choose which classifier you’d like to use. You can use
# scikit learn for this.
def train_classifier(train_data_file: str, vocab):
    pass

# This function takes the trained classifier, a test image and the vocabulary as inputs. It generates the feature vector
# for the test image using the vocabulary and runs it through the classifier, returning the output classification.
def classify_image(classifier, test_img, vocabulary):
    pass

# This function will take an image and two thresholds and perform hysteresis thresholding, producing a black and
# white image.
def threshold_image(image, low_thresh, high_thresh):
    pass

# This function will take an image as input. Use one of the techniques from class to perform region growing,
# returning the output region map.
def grow_regions(image):
    pass

# This function will take an image as input. Use one of the techniques from class to perform region splitting,
# returning the output region map.
def split_regions(image):
    pass

# This function will take an image as input. Use one of the techniques from class to perform region merging,
# returning the output region map
def merge_regions(image):
    pass


# This function will take an image as input. Using different combinations of the above methods, extract three
# segmentation maps with labels to indicate the approach.
def segment_image(image):
    pass

# Use Kmeans to perform image segmentation. You’re free to do this however you’d like. Do not assume the number
# of classes is 2. So you’ll want to implement a method for determining what k should be. Provide details in your
# README.txt.
def kmeans_segment(image):
    pass



