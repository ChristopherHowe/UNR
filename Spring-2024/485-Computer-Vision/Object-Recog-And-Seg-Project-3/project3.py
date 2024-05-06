import cv2
import numpy as np
import math
from typing import List, Dict
from sklearn.linear_model import Perceptron
from queue import Queue


def load_img(file_name):
    return cv2.imread(file_name)


def display_img(image: np.ndarray):
    cv2.imshow("image filtering project 3", image)
    k = cv2.waitKey(0)
    cv2.destroyAllWindows()


def generate_vocabulary(train_data_file: str):
    print("Calling generate vocabulary")
    vocabulary = np.zeros((0, 128))
    with open(train_data_file, "r") as file:
        sift = cv2.SIFT_create()
        sift.setNFeatures(100)

        for line in file:
            img_name = line.split()[0]
            img = load_img(img_name)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, des = sift.detectAndCompute(gray, None)
            vocabulary = np.concatenate((vocabulary, des))
    return vocabulary


def extract_features(image: np.ndarray, vocabulary: np.ndarray):
    print("Calling extract_features")
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

    return bowVect


image_num = 0


def train_classifier(train_data_file: str, vocab: np.ndarray):
    print("Calling train_classifier")
    with open(train_data_file, "r") as file:
        sift = cv2.SIFT_create()
        sift.setNFeatures(100)

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


def classify_image(classifier: Perceptron, test_img: np.ndarray, vocabulary: np.ndarray):
    print("Calling classify_image")
    bow_vect = extract_features(test_img, vocabulary)
    prediction = classifier.predict(bow_vect.reshape(1, -1))
    if prediction.ndim != 1 or prediction.size != 1:
        raise ValueError(f"Classifier prediction is not in the expected shape (shape:{prediction.shape})")
    return prediction


def euclidean_distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def threshold_image(image, low_thresh, high_thresh):
    print("Calling threshold_image")
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
    return np.uint8(result * 255)


class Region:
    def __init__(self, seed, color):
        self.points = [seed]
        self.grow_queue = [seed]
        self.color = color


def getRegionMap(regions: List[Region], shape):
    newImg = np.zeros(shape, dtype=np.uint8)
    for region in regions:
        for point in region.points:
            newImg[point] = region.color
    return newImg


def getNeighbors(point):
    return [(point[0] + 1, point[1]), (point[0], point[1] + 1), (point[0] - 1, point[1]), (point[0], point[1] - 1)]


def grow_regions(image: np.ndarray):
    print("Calling grow_regions")

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
                for newPoint in getNeighbors(point):
                    if newPoint not in assigned and 0 < newPoint[0] < img_w and 0 < newPoint[1] < img_h:
                        converged = False
                        areSim = areSimilar(point, newPoint)
                        if areSim:
                            region.points.append(newPoint)
                            assigned.add(newPoint)
                            new_grow_queue.append(newPoint)
            region.grow_queue = new_grow_queue
    return getRegionMap(regions, image.shape)


def split_regions(image: np.ndarray):
    print("Calling split_regions")

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


class RAG:
    def __init__(self):
        self.nodes: Dict[int, set[int]] = {}

    def addNode(self, val: int):
        if val in self.nodes:
            raise ValueError(f"{val} is already in nodes")
        self.nodes[val] = set()

    def addEdge(self, n1: int, n2: int):
        if n1 not in self.nodes or n2 not in self.nodes:
            raise ValueError(f"{n1} or {n2} is not in nodes")
        if n2 not in self.nodes[n1]:
            self.nodes[n1].add(n2)
            self.nodes[n2].add(n1)

    def mergeNodes(self, n1: int, n2: int):
        if n1 not in self.nodes or n2 not in self.nodes:
            raise ValueError(f"{n1} or {n2} is not in nodes")
        self.nodes[n1].union(self.nodes[n2])
        if n1 in self.nodes[n1]:
            self.nodes[n1].remove(n1)
        for node, edges in self.nodes.items():
            if n2 in edges:
                edges.remove(n2)
                if node != n1:
                    edges.add(n1)
        del self.nodes[n2]


def merge_regions(image: np.ndarray):
    print("Calling merge_regions")

    class MergeRegion:
        def __init__(self, color, id):
            self.points = set()
            self.color = color
            self.id: int = id

    def getRegionMap(regions: set[MergeRegion], image):
        newImg = np.copy(image)
        for region in regions:
            for point in region.points:
                newImg[point] = region.color
        return newImg

    def getCurrrentRegions():
        def getPointsRegion(p):
            for region in regions:
                for point in region.points:
                    if point == p:
                        return region.id

        regions: list[MergeRegion] = []
        regionCounter = 0
        rag = RAG()
        visited = set()
        for imgPoint, _ in np.ndenumerate(image[..., 0]):
            if imgPoint not in visited:
                visited.add(imgPoint)
                newRegion = MergeRegion(image[imgPoint], regionCounter)
                rag.addNode(regionCounter)
                newRegion.points.add(imgPoint)
                grow_queue = Queue()
                grow_queue.put(imgPoint)

                while not grow_queue.empty():
                    p = grow_queue.get()
                    neighbors = getNeighbors(p)
                    for neighbor in neighbors:
                        if 0 <= neighbor[0] < image.shape[0] and 0 <= neighbor[1] < image.shape[1]:
                            if neighbor not in visited:
                                if (image[neighbor] == image[imgPoint]).all():
                                    newRegion.points.add(neighbor)
                                    grow_queue.put(neighbor)
                                    visited.add(neighbor)
                            elif neighbor not in newRegion.points:
                                rag.addEdge(getPointsRegion(neighbor), regionCounter)

                regions.append(newRegion)
                regionCounter += 1

        return regions, rag

    regions, rag = getCurrrentRegions()

    def mergeRegions(r1: MergeRegion, r2: MergeRegion):
        r1.points = r1.points.union(r2.points)
        total_points = len(r1.points) + len(r2.points)
        weighted_color_sum = len(r1.points) * np.float64(r1.color) + len(r2.points) * np.float64(r2.color)
        r1.color = np.uint8(weighted_color_sum / total_points)

    def getRegionToMerge(r: RAG):
        for id, edges in r.nodes.items():
            r1 = regions[id]
            for edge in edges:
                r2 = regions[edge]
                diff = np.sum(abs(np.int32(r1.color) - np.int32(r2.color)))
                if diff <= 40:
                    return r1, r2
        return (None, None)

    while True:
        (r1, r2) = getRegionToMerge(rag)
        if r1 is None and r2 is None:
            break
        mergeRegions(r1, r2)
        rag.mergeNodes(r1.id, r2.id)

    return getRegionMap(regions, image)


def segment_image(image):
    print("Calling segment_image")
    thresheld = threshold_image(image, 40, 100)
    grown = grow_regions(image)
    merged = merge_regions(split_regions(merge_regions(split_regions(image))))
    return thresheld, grown, merged
