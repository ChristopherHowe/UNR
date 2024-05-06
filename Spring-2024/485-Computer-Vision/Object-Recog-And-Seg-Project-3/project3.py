import cv2
import numpy as np
import math
from typing import List
from sklearn.linear_model import Perceptron
import matplotlib.pyplot as plt
from queue import Queue


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


def generate_vocabulary(train_data_file: str):
    # Reading file line by line
    vocabulary = np.zeros((0, 128))
    with open(train_data_file, "r") as file:
        sift = cv2.SIFT_create()
        sift.setNFeatures(100)

        for line in file:
            img_name = line.split()[0]
            img = load_img(img_name)
            # display_img(img)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, des = sift.detectAndCompute(gray, None)
            vocabulary = np.concatenate((vocabulary, des))
    return vocabulary


def extract_features(image: np.ndarray, vocabulary: np.ndarray):
    sift = cv2.SIFT_create()
    sift.setNFeatures(100)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, descriptors = sift.detectAndCompute(gray, None)
    bowVect = np.zeros(vocabulary.shape[0])
    descriptorInd = 0
    for descriptor in descriptors:
        differences = np.sum((vocabulary - descriptor) ** 2, axis=1)
        closestWordInd = np.argmin(differences)
        bowVect[closestWordInd] += 1

        descriptorInd += 1
    # print(f"extracting length of bow vect: {np.sum(bowVect)}")

    return bowVect


image_num = 0


def train_classifier(train_data_file: str, vocab: np.ndarray):
    with open(train_data_file, "r") as file:
        sift = cv2.SIFT_create()
        sift.setNFeatures(100)

        # Perceptron setup
        # - Setting random state seed to zero makes every run the same.
        # - eta0 is the initial learning rate of the model.
        # - max iterations is set to prevent the model from running indefinitely or overfitting to the data.
        p = Perceptron(max_iter=100, eta0=0.1, random_state=0)

        training_bow_vectors = np.zeros((0, vocab.shape[0]))
        training_classifications = np.array([])

        for line in file:
            global image_num

            line_parts = line.split()
            img_name = line_parts[0]
            img_classification = int(line_parts[1])
            img = load_img(img_name)
            bowVect = extract_features(img, vocab)
            training_bow_vectors = np.concatenate((training_bow_vectors, bowVect[np.newaxis, :]))

            training_classifications = np.append(training_classifications, img_classification)
            image_num += 1

        print("Fitting training data")
        p.fit(training_bow_vectors, training_classifications)
        return p


# This function takes the trained classifier, a test image and the vocabulary as inputs. It generates the feature vector
# for the test image using the vocabulary and runs it through the classifier, returning the output classification.
def classify_image(classifier: Perceptron, test_img: np.ndarray, vocabulary: np.ndarray):
    bow_vect = extract_features(test_img, vocabulary)
    prediction = classifier.predict(bow_vect.reshape(1, -1))
    if prediction.ndim != 1 or prediction.size != 1:
        raise ValueError(f"Classifier prediction is not in the expected shape (shape:{prediction.shape})")
    return prediction


def euclidean_distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


# This function will take an image and two thresholds and perform hysteresis thresholding, producing a black and
# white image.
def threshold_image(image, low_thresh, high_thresh):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_w = image.shape[0]
    img_h = image.shape[1]

    def threshold(img: np.ndarray, threshold: int):
        new_img = np.where(img < threshold, 0, 1)
        return new_img

    def neighbor_not_zero(img_x, img_y):
        for i in range(-1, 2):
            for j in range(-1, 2):
                if 0 <= img_x + i < img_w and 0 <= img_y + j < img_h and high_t_img[img_x + i][img_y + j] != 0:
                    return 1
        return 0

    low_t_img = threshold(gray, low_thresh)
    high_t_img = threshold(gray, high_thresh)

    result = np.copy(high_t_img)

    for img_x in range(img_w):
        for img_y in range(img_h):
            if low_t_img[img_x][img_y] != 0:
                result[img_x][img_y] = neighbor_not_zero(img_x, img_y)
    return np.uint8(result)


class Region:
    def __init__(self, seed, color):
        self.points = set()
        self.points.add(seed)
        self.grow_queue = [seed]
        self.color = color


def getRegionMap(regions: List[Region], shape):
    newImg = np.zeros(shape)
    for region in regions:
        for point in region.points:
            newImg[point] = region.color
    return newImg


# This function will take an image as input. Use one of the techniques from class to perform region growing,
# returning the output region map.
def grow_regions(image: np.ndarray):
    def getSeedValues(image: np.ndarray):
        # get a histogram showing the frequencies for each possible pixel value.
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        flat_img = gray.flatten()
        hist, _ = np.histogram(flat_img, bins=range(256))
        # Generate a list of peaks of the histogram by finding all values that are higher than either of their neighbors
        peaks = np.where((hist[:-2] < hist[1:-1]) & (hist[1:-1] > hist[2:]))[0] + 1
        seeds = []

        # only consider seeds that are not near any of the other seeds
        def nearAnotherSeed(newSeed, threshold):
            for seed in seeds:
                if euclidean_distance(seed, newSeed) < threshold:
                    return True
            return False

        for peak in peaks:
            for index, value in np.ndenumerate(gray):
                if value == peak and not nearAnotherSeed(index, image.shape[1] / 15):
                    seeds.append(index)
                    break

        return seeds

    def areSimilar(p1, p2):
        return np.sum(abs(subtractable_image[p1[0], p1[1]] - subtractable_image[p2[0], p2[1]])) < 60

    seeds = getSeedValues(image)
    regions: List[Region] = []
    assigned = set()

    for seed in seeds:
        regions.append(Region(seed, np.random.randint(0, 256, size=3, dtype=np.uint8)))
        assigned.add(seed)

    img_w = image.shape[0]
    img_h = image.shape[1]
    subtractable_image = np.int16(image)

    # Actual region growing implementation
    # Inspired by this stack overflow question https://stackoverflow.com/a/5851382
    converged = False
    while not converged:
        converged = True
        for region in regions:
            new_grow_queue = []
            for point in region.grow_queue:
                for dir in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    newPoint = (point[0] + dir[0], point[1] + dir[1])
                    if newPoint not in assigned and 0 < newPoint[0] < img_w and 0 < newPoint[1] < img_h:
                        converged = False
                        areSim = areSimilar(point, newPoint)
                        if areSim:
                            region.points.add(newPoint)
                            assigned.add(newPoint)
                            new_grow_queue.append(newPoint)
            region.grow_queue = new_grow_queue

    return getRegionMap(regions, image.shape)


# This function will take an image as input. Use one of the techniques from class to perform region splitting,
# returning the output region map.
def split_regions(image: np.ndarray):
    def split_recursive(block: np.ndarray, depth):
        def isBlockUniform(b):
            std = (np.std(b[:, :, 0]) + np.std(b[:, :, 1]) + np.std(b[:, :, 2])) / 3
            return std < 15

        rows, cols, _ = block.shape
        if not isBlockUniform(block) and block.shape[0] >= 2 and block.shape[1] >= 2:
            # Uses integer division to ensure valid splitting operations
            block[: rows // 2, : cols // 2] = split_recursive(block[: rows // 2, : cols // 2], depth + 1)  # Top left
            block[: rows // 2, cols // 2 :] = split_recursive(block[: rows // 2, cols // 2 :], depth + 1)  # top right
            block[rows // 2 :, : cols // 2] = split_recursive(block[rows // 2 :, : cols // 2], depth + 1)  # Bottom Left
            block[rows // 2 :, cols // 2 :] = split_recursive(block[rows // 2 :, cols // 2 :], depth + 1)  # Bottom Right
        else:
            block = np.array([np.mean(block[:, :, 0]), np.mean(block[:, :, 1]), np.mean(block[:, :, 2])])
        return block

    return split_recursive(np.copy(image), 1)


# This function will take an image as input. Use one of the techniques from class to perform region merging,
# returning the output region map
def merge_regions(image):
    regions = List[set] = List(set())
    for (point)


# This function will take an image as input. Using different combinations of the above methods, extract three
# segmentation maps with labels to indicate the approach.
def segment_image(image):
    # thresheld = threshold_image(image, 40, 100)
    # grown = grow_regions(image)
    # This page was used for learning about splitting and merging
    # https://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/MARBLE/medium/segment/split.htm
    split = split_regions(image)
    display_img(split)
    exit()
    # return thresheld, grown, split


# Use Kmeans to perform image segmentation. You’re free to do this however you’d like. Do not assume the number
# of classes is 2. So you’ll want to implement a method for determining what k should be. Provide details in your
# README.txt.
def kmeans_segment(image):
    pass
