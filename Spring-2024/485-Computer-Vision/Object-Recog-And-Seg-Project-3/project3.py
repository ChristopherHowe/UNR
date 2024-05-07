import cv2
import numpy as np
import math
from typing import List, Dict
from sklearn.linear_model import Perceptron
from queue import Queue
import time


def load_img(file_name):
    return cv2.imread(file_name)


def display_img(image: np.ndarray):
    cv2.imshow("image filtering project 3", image)
    k = cv2.waitKey(0)
    cv2.destroyAllWindows()


def k_means(points: np.ndarray, k: int):
    # assumes each points has n points each of with length m so points.shape = (n,m)
    max_center_movement = 0.01  # How much each center can move and still be considered stable

    def distance(p1: np.ndarray, p2: np.ndarray):
        return np.sum((np.float64(p1) - np.float64(p2)) ** 2)

    def getClosestPoint(p1, points):
        closest_point_ind = 0
        closest_point_val = distance(p1, points[0])
        for point_ind in range(1, len(points)):
            d = distance(p1, points[point_ind])
            if d < closest_point_val:
                closest_point_ind = point_ind
                closest_point_val = d
        return closest_point_ind

    # Get some random centers at the beggining
    rng = np.random.default_rng()
    centers = rng.choice(points, size=k, replace=False)

    converged = False
    while not converged:
        # Set up the clusters
        clusters: List[List[np.ndarray]] = []
        for _ in centers:
            clusters.append([])
        # assign each datapoint to a cluster
        for point in points:
            closest_center_ind = getClosestPoint(point, centers)
            clusters[closest_center_ind].append(point)
        new_centers = np.zeros(shape=centers.shape)
        # recalculate the cluster centers, simultaneously check for convergence.
        converged = True
        for cluster_ind, cluster in enumerate(clusters):
            new_center: np.ndarray = np.average(np.array(cluster), axis=0)
            new_centers[cluster_ind] = new_center
            if distance(new_center, centers[cluster_ind]) >= max_center_movement:
                converged = False
        centers = new_centers
    return centers


def generate_vocabulary(train_data_file: str):
    print("Calling generate vocabulary")
    all_features = np.zeros((0, 9))  # HOG features with bins of size 40 degrees of gradient orientation
    with open(train_data_file, "r") as file:
        for line in file:
            img_name = line.split()[0]
            img = load_img(img_name)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_feat_descriptors = getAllDescriptors(gray)
            all_features = np.concatenate((all_features, img_feat_descriptors))

    print("running k means on vocab")
    vocabulary = k_means(all_features, 120)
    return vocabulary


def extract_features(image: np.ndarray, vocabulary: np.ndarray):
    print("Calling extract_features")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    bowVect = np.zeros(vocabulary.shape[0])
    for descriptor in getAllDescriptors(gray):
        differences = np.sum((vocabulary - descriptor) ** 2, axis=1)
        closestWordInd = np.argmin(differences)
        bowVect[closestWordInd] += 1
    return bowVect


def train_classifier(train_data_file: str, vocab: np.ndarray):
    print("Calling train_classifier")
    with open(train_data_file, "r") as file:
        p = Perceptron(max_iter=100, eta0=0.1, random_state=0)

        training_bow_vectors = np.zeros((0, vocab.shape[0]))
        training_classifications = np.array([])

        for line in file:
            line_parts = line.split()
            img_name = line_parts[0]
            img_classification = int(line_parts[1])
            img = load_img(img_name)
            bowVect = extract_features(img, vocab)

            training_bow_vectors = np.concatenate((training_bow_vectors, bowVect[np.newaxis, :]))
            training_classifications = np.append(training_classifications, img_classification)

        print("Fitting training data")
        p.fit(training_bow_vectors, training_classifications)
        return p


def classify_image(classifier: Perceptron, test_img: np.ndarray, vocabulary: np.ndarray):
    print("Calling classify_image")
    bow_vect = extract_features(test_img, vocabulary)
    return classifier.predict(bow_vect.reshape(1, -1))


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


#######################################
## Feature Extraction From Project 2 ##
#######################################


def getAllDescriptors(img: np.ndarray):
    print("calling getAllDescriptors")
    descriptors = []
    print("Getting keypoints")
    detected_keypoints = harris_detector(img)
    print("Extracting HOG features")
    for keypoint in detected_keypoints:
        descriptors.append(extract_HOG(img, keypoint))
    return descriptors


def harris_detector(image: np.ndarray):
    WINDOW_SIZE = 3
    THTRESHOLD_PERCENT = 80
    K_VAL = 0.005

    img_w = image.shape[0]
    img_h = image.shape[1]
    padding_size: int = math.floor(WINDOW_SIZE / 2)

    new_img: np.ndarray
    if image.ndim == 3:
        new_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        new_img = np.copy(image)
    padded_src, pad_vals = pad_img(new_img, padding_size, 0)

    sobel_kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    sobel_kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

    Mx = apply_filter(padded_src, sobel_kernel_x, 1, 1)
    My = apply_filter(padded_src, sobel_kernel_y, 1, 1)
    Mxx = apply_filter(Mx, sobel_kernel_x, 1, 1)
    Myy = apply_filter(My, sobel_kernel_y, 1, 1)
    Mxy = apply_filter(Mx, sobel_kernel_y, 1, 1)

    start_time = time.time()
    Aw = np.stack((Mxx, Mxy, Mxy, Myy), axis=2).reshape(Mx.shape[0], Mx.shape[1], 2, 2)
    R_new = np.linalg.det(Aw) - K_VAL * np.trace(Aw, axis1=-2, axis2=-1)
    detected_new = np.copy(R_new)
    detected_new[detected_new < 0] = 0
    end_time = time.time()
    duration = end_time - start_time
    print("took", duration, "seconds with numpy implementation")

    start_time = time.time()
    detected = np.zeros_like(Mx)
    for img_x in range(padding_size, img_w - padding_size):
        for img_y in range(padding_size, img_h - padding_size):
            Aw_xy = Aw[img_x][img_y]
            R = np.linalg.det(Aw_xy) - K_VAL * (np.trace(Aw_xy) ** 2)
            # if R < 0 edge, if R ~ small -> flat, if R > 0 and big -> corner
            #### dropping flats is handled by thresholding, dropping edges handled by max func
            detected[img_x][img_y] = max(0, R)
    end_time = time.time()
    duration = end_time - start_time
    print("took", duration, "seconds with looped implementation")

    unpadded = unpad_img(detected, pad_vals)
    suppressed = non_maxima_suppression(unpadded)
    thresheld = normalizedThreshold(suppressed, THTRESHOLD_PERCENT)
    return binary_img_to_point_arr(thresheld)


def extract_HOG(image: np.ndarray, keypoint):
    WINDOW_SIZE = 16
    BIN_DEG_RANGE = 40
    if 360 % BIN_DEG_RANGE != 0:
        raise ValueError("BIN_DEG_RANGE does not evenly divide 360")

    def rad_to_bin(rad) -> int:
        return math.floor(((math.degrees(rad) + 180) % 360) / BIN_DEG_RANGE)

    num_bins = int(360 / BIN_DEG_RANGE)
    offset = math.floor(WINDOW_SIZE / 2)
    histogram = np.zeros(num_bins)
    gradient = make_gradiant(safe_slice(image, keypoint.x - offset, keypoint.x + offset + 1, keypoint.y - offset, keypoint.y + offset + 1), 3)

    magnitude = gradient[:, :, 0]
    orientation = gradient[:, :, 1]

    for wx in range(0, WINDOW_SIZE):
        for wy in range(0, WINDOW_SIZE):
            assigned_bin = rad_to_bin(orientation[wx, wy])
            histogram[assigned_bin] += magnitude[wx, wy]
    return histogram


# assumes that the src has single value data (not rgb)
def non_maxima_suppression(src: np.ndarray):
    TILE_SIZE = 15
    TOO_CLOSE_RADIUS = TILE_SIZE / 2
    NUM_IPOINTS_PER_TILE = TILE_SIZE

    # Found the idea for this implementation here https://www.ipol.im/pub/art/2018/229/article_lr.pdf
    def suppress_tile_non_maxima(tile: np.ndarray) -> np.ndarray:
        def distanceFromPoint(p1, p2):
            return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

        def too_close_to_others(p1, others, r) -> bool:
            for other in others:
                if distanceFromPoint(p1, other) < r:
                    return True
            return False

        tile_sorted_inds = np.argsort(tile.flatten())
        sortedInds = []
        for ind in tile_sorted_inds:
            pos = np.unravel_index(ind, tile.shape)
            sortedInds.append(pos)
        filtered_sorted_inds = []
        for ind in reversed(sortedInds):
            if tile[ind[0]][ind[1]] != 0:
                filtered_sorted_inds.append(ind)
            else:
                break
        accepted_corners = []
        resulting_tile = np.zeros(tile.shape, dtype=tile.dtype)
        for ind in filtered_sorted_inds:
            if too_close_to_others(ind, accepted_corners, TOO_CLOSE_RADIUS):
                continue
            accepted_corners.append(ind)
            resulting_tile[ind[0]][ind[1]] = tile[ind[0]][ind[1]]
            if len(accepted_corners) >= NUM_IPOINTS_PER_TILE:
                break
        return resulting_tile

    src_w = src.shape[0]
    src_h = src.shape[1]

    # Check that src has intensity data (not rgb)
    if src.ndim != 2:
        raise ValueError("src does not have two dimensions, make sure you are not passing it intensity data")
    # tiles can be any size less than the tile size (edges can be less if corners size is not evenly divided by tile size)
    result = np.zeros(src.shape, dtype=src.dtype)

    def get_num_tiles(length, tile_size) -> int:
        if length % tile_size == 0:
            return int(length / tile_size)
        else:
            return math.floor(src_w / tile_size) + 1

    num_tiles_wide = get_num_tiles(src_w, TILE_SIZE)
    num_tiles_tall = get_num_tiles(src_h, TILE_SIZE)

    for tile_x in range(0, num_tiles_wide):
        tile_x_start = tile_x * TILE_SIZE
        tile_x_end = min((tile_x + 1) * TILE_SIZE, src_w)
        for tile_y in range(0, num_tiles_tall):
            tile_y_start = tile_y * TILE_SIZE
            tile_y_end = min((tile_y + 1) * TILE_SIZE, src_h)
            src_tile = src[tile_x_start:tile_x_end, tile_y_start:tile_y_end]
            result_tile = suppress_tile_non_maxima(src_tile)
            result[tile_x_start:tile_x_end, tile_y_start:tile_y_end] = result_tile

    return result


def pad_img(image, size, val):
    pad_values = ()
    if image.ndim == 3:
        pad_values = ((size, size), (size, size), (0, 0))
    elif image.ndim == 2:
        pad_values = ((size, size), (size, size))
    new_img = np.zeros(image.shape, dtype=np.uint8)
    if val == 0:
        new_img = np.pad(image, pad_values, mode="constant", constant_values=0)
    else:
        new_img = np.pad(image, pad_values, mode="edge")
    return new_img, pad_values


def unpad_img(src, padded_vals):
    slices = []
    for c in padded_vals:
        e = None if c[1] == 0 else -c[1]
        slices.append(slice(c[0], e))
    return src[tuple(slices)]


def threshold(img: np.ndarray, threshold: int):
    new_img = np.where(img < threshold, 0, 1)
    return new_img


def normalizedThreshold(img: np.ndarray, percent):
    sorted_minvals = np.sort(img[img != 0].flatten())
    normalized_t = sorted_minvals[int(len(sorted_minvals) * (percent / 100))]
    return threshold(img, normalized_t)


def safe_slice(arr: np.ndarray, x1, x2, y1, y2):
    arr_w = arr.shape[0]
    arr_h = arr.shape[1]
    if x1 < 0 or x2 > arr_w or y1 < 0 or y2 > arr_h:
        l_pad = max(0 - x1, 0)
        r_pad = max(x2 - arr_w, 0)
        t_pad = max(0 - y1, 0)
        b_pad = max(y2 - arr_h, 0)
        padding = ((l_pad, r_pad), (t_pad, b_pad))
        if arr.ndim == 3:
            padding = ((l_pad, r_pad), (t_pad, b_pad), (0, 0))
        padded_arr = np.pad(arr, padding, mode="edge")
        return padded_arr[x1 + l_pad : x2 + l_pad, y1 + t_pad : y2 + t_pad]
    return arr[x1:x2, y1:y2]


def apply_filter(image: np.ndarray, mask: np.ndarray, pad_pixels: int, pad_value: int):
    def correlation(image: np.ndarray, mask: np.ndarray, img_x: int, img_y: int):
        val: int = 0
        for mask_x in range(mask_w):
            for mask_y in range(mask_h):
                x_diff = int(mask_x - (mask_w / 2) + 0.5)
                y_diff = int(mask_y - (mask_h / 2) + 0.5)
                src_val = np.mean(image[img_x + x_diff][img_y + y_diff])
                step = src_val * mask[mask_x][mask_y]
                val += step
        return val

    def handle_mask_check(mask: np.ndarray):
        if mask.ndim == 1:  # make 1D arrays into 2D with width 1
            mask = mask.reshape(1, -1)
        if mask.ndim > 2:
            raise ValueError("Does not support masks with a higher dimension than 2")
        return mask

    mask = handle_mask_check(mask)
    mask_w = mask.shape[0]
    mask_h = mask.shape[1] if mask.ndim == 2 else 0
    req_w_space = math.floor(mask_w / 2)
    req_h_space = math.floor(mask_h / 2)

    src, pad_values = pad_img(image, pad_pixels, pad_value)
    src_w = src.shape[0]
    src_h = src.shape[1]

    new_img = np.zeros(src.shape, dtype=np.float64)
    for img_x in range(req_w_space, src_w - req_w_space):
        for img_y in range(req_h_space, src_h - req_h_space):
            v = correlation(src, mask, img_x, img_y)
            new_img[img_x][img_y] = v

    new_img = unpad_img(new_img, pad_values)
    return new_img


def generate_gaussian(sigma, filter_w, filter_h):
    def gaussian_func_2d(x, y, sigma):
        base = 1 / (2 * math.pi * sigma * sigma)
        exponent = -1 * ((x * x + y * y) / (2 * sigma * sigma))
        result = base * math.exp(exponent)
        return result

    if filter_w < 1 or filter_h < 1:
        raise ValueError("filter height and width must be 1 or greater")

    mask = np.zeros((filter_w, filter_h))
    for x in range(0, filter_w):
        for y in range(0, filter_h):
            x_val = x - (filter_w / 2) + 0.5
            y_val = y - (filter_h / 2) + 0.5
            v = gaussian_func_2d(x_val, y_val, sigma)
            mask[x][y] = v
    normalized_mask = mask / np.sum(mask)
    return normalized_mask


def make_gradiant(img: np.ndarray, gaussian_size: int):
    img_w = img.shape[0]
    img_h = img.shape[1]
    sigma = gaussian_size / 5
    gaussian = generate_gaussian(sigma, gaussian_size, gaussian_size)
    horizontal_kernel = np.array([[-1, 0, 1]])
    vertical_kernel = np.array([[-1], [0], [1]])
    x_deriv_gaussian = apply_filter(gaussian, horizontal_kernel, 1, 1)
    y_deriv_gaussian = apply_filter(gaussian, vertical_kernel, 1, 1)

    new_img: np.ndarray
    if img.ndim == 3:
        new_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        new_img = np.copy(img)
    Mx = apply_filter(new_img, x_deriv_gaussian, math.floor(gaussian_size / 2), 1)
    My = apply_filter(new_img, y_deriv_gaussian, math.floor(gaussian_size / 2), 1)
    gradient = np.zeros((img_w, img_h, 2), dtype=np.float64)
    for img_x in range(img_w):
        for img_y in range(img_h):
            magnitude = math.sqrt(math.pow(Mx[img_x][img_y], 2) + math.pow(My[img_x][img_y], 2))
            angle = math.atan2(Mx[img_x][img_y], My[img_x][img_y])
            gradient[img_x][img_y] = [magnitude, angle]
    return gradient


class point:
    def __init__(self, x, y):
        self.x: int = x
        self.y: int = y


def binary_img_to_point_arr(image: np.ndarray) -> List[point]:
    points: List[point] = []
    for x in range(0, image.shape[0]):
        for y in range(0, image.shape[1]):
            if image[x][y] == 1:
                points.append(point(x, y))
    return points
