import cv2
import numpy as np
import math
from typing import List


def load_img(file_name):
    return cv2.imread(file_name)


def display_img(image: np.ndarray):
    cv2.imshow("image filtering project 1", image)
    k = cv2.waitKey(0)
    cv2.destroyAllWindows()


def display_img_normalized(image: np.ndarray):
    # normalized_image = np.zeros(image.shape, dtype=np.uint8)
    # cv2.normalize(image, normalized_image, 0,255,cv2.NORM_MINMAX)
    normalized_image = np.uint8(
        (image - np.min(image)) / (np.max(image) - np.min(image)) * 255
    )
    cv2.imshow("image filtering project 1", normalized_image)
    k = cv2.waitKey(0)
    cv2.destroyAllWindows()


def aspect_scale(img: np.ndarray, height):
    ratio = img.shape[0] / img.shape[1]
    return cv2.resize(img, (height, int(height * ratio)))


class point:
    def __init__(self, x, y):
        self.x: int = x
        self.y: int = y


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


def unpad_img(src, pad_width):
    # this unpad function comes from this stack overflow question https://stackoverflow.com/a/57956349
    slices = []
    for c in pad_width:
        e = None if c[1] == 0 else -c[1]
        slices.append(slice(c[0], e))
    return src[tuple(slices)]


def threshold(img: np.ndarray, threshold: int):
    new_img = np.where(img < threshold, 0, 1)
    return new_img


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
        raise ValueError(
            "src does not have two dimensions, make sure you are not passing it intensity data"
        )
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


def rgb2gray(rgb: np.ndarray):
    grey = np.zeros(rgb.shape[:-1], dtype=np.uint8)
    rgb_w = rgb.shape[0]
    rgb_h = rgb.shape[1]
    for rgb_x in range(0, rgb_w):
        for rgb_y in range(0, rgb_h):
            grey[rgb_x][rgb_y] = int(sum(rgb[rgb_x][rgb_y]) / 3)
    return grey


def binary_img_to_point_arr(image: np.ndarray) -> List[point]:
    points: List[point] = []
    for x in range(0, image.shape[0]):
        for y in range(0, image.shape[1]):
            if image[x][y] == 1:
                points.append(point(x, y))
    return points


# Currently assumes all channels are equal (grey scale)
def moravec_detector(image: np.ndarray) -> List[point]:
    WINDOW_SIZE: int = 3
    THRESHOLD = 80
    PRINC_DIRS = [
        point(0, -1),
        point(0, 1),
        point(-1, 0),
        point(-1, -1),
        point(-1, 1),
        point(1, 0),
        point(1, -1),
        point(1, 1),
    ]

    local_src = np.copy(image)
    if image.ndim == 3:
        local_src = rgb2gray(image)

    img_w = local_src.shape[0]
    img_h = local_src.shape[1]

    padding_size: int = (
        math.floor(WINDOW_SIZE / 2) + 1
    )  # plus val should be equal to princ dir len
    padded_src, padVals = pad_img(local_src, padding_size, 0)

    def getSW(image: np.ndarray, x: int, y: int, princDir: point):
        sumDiffs = 0
        for i in range(WINDOW_SIZE):
            window_x: int = x + i - math.floor(WINDOW_SIZE / 2)
            for j in range(WINDOW_SIZE):
                window_y = y + j - math.floor(WINDOW_SIZE / 2)
                anchorPx = int(image[window_x][window_y])
                shiftedPx = int(image[window_x - princDir.x][window_y - princDir.y])
                pixel_diff = anchorPx - shiftedPx
                sumDiffs += pow(pixel_diff, 2)
        return sumDiffs

    new_img = np.zeros(padded_src.shape, dtype=np.int32)
    for img_x in range(padding_size, img_w - padding_size):
        for img_y in range(padding_size, img_h - padding_size):
            minVal = 100000
            for princDir in PRINC_DIRS:
                Sw = getSW(padded_src, img_x, img_y, princDir)
                minVal = min(minVal, Sw)
            new_img[img_x][img_y] = minVal

    def normalizedThreshold(val_img: np.ndarray, percent):
        sorted_minvals = np.sort(val_img[val_img != 0].flatten())
        return sorted_minvals[int(len(sorted_minvals) * (percent / 100))]

    unpadded_img = unpad_img(new_img, padVals)
    suppressed_img = non_maxima_suppression(unpadded_img)
    thresheld_img = threshold(
        suppressed_img, normalizedThreshold(suppressed_img, THRESHOLD)
    )
    return binary_img_to_point_arr(thresheld_img)


# def harris_detector(image):


# Expects image to be in color
def plot_keypoints(image, keypoints: List[point]):
    def markPx(image, x, y):
        MARK_SIZE = 5
        pattern = [
            [0, 1, 1, 1, 0],
            [1, 0, 0, 0, 1],
            [1, 0, 1, 0, 1],
            [1, 0, 0, 0, 1],
            [0, 1, 1, 1, 0],
        ]
        for lx in range(0, MARK_SIZE):
            for ly in range(0, MARK_SIZE):
                if (
                    pattern[lx][ly] == 1
                    and x + lx - 2 < image.shape[0]
                    and y + ly - 2 < image.shape[1]
                ):
                    image[x + lx - 2][y + ly - 2] = [0, 0, 255]

    result = np.copy(image)
    for keypoint in keypoints:
        markPx(result, keypoint.x, keypoint.y)
    return result


def extract_LBP(image, keypoint: point):
    def get_binary_pattern_as_base_10(image, x, y):
        threshold = image[x][y]
        val = 0
        for wx in range(-1, 2):
            for wy in range(-1, 2):
                if wx == 0 and wy == 0:
                    continue
                val = val * 2 + (1 if image[x + wx][y + wy] > threshold else 0)
        return val

    local_src = np.copy(image)
    if image.ndim == 3:
        local_src = rgb2gray(image)

    histogram = np.zeros(256)
    WINDOW_SIZE = 16
    img_w = local_src.shape[0]
    img_h = local_src.shape[1]
    for wx in range(0, WINDOW_SIZE):
        img_x = keypoint.x + wx - int(WINDOW_SIZE / 2) - 1
        for wy in range(0, WINDOW_SIZE):
            img_y = keypoint.y + wy - int(WINDOW_SIZE / 2) - 1
            if (img_x not in range(1, img_w - 1)) or (img_y not in range(1, img_h - 1)):
                histogram[0] += 1
            else:
                binary_pattern = get_binary_pattern_as_base_10(local_src, img_x, img_y)
                if binary_pattern not in range(0, 256):
                    raise ValueError(
                        "Something went wrong extracting binary pattern, value is not 0-255"
                    )
                histogram[binary_pattern] += 1
    return histogram


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

    def handle_padding(image):
        pad_values = ()
        if image.ndim == 3:
            pad_values = ((pad_pixels, pad_pixels), (pad_pixels, pad_pixels), (0, 0))
        elif image.ndim == 2:
            pad_values = ((pad_pixels, pad_pixels), (pad_pixels, pad_pixels))

        if pad_value == 0:
            image = np.pad(image, pad_values, mode="constant", constant_values=0)
        else:
            image = np.pad(image, pad_values, mode="edge")
        return image, pad_values

    def handle_unpadding(image, prev_padding):
        def unpad(x, pad_width):
            # this unpad function comes from this stack overflow question https://stackoverflow.com/a/57956349
            slices = []
            for c in pad_width:
                e = None if c[1] == 0 else -c[1]
                slices.append(slice(c[0], e))
            return x[tuple(slices)]

        if pad_pixels >= req_w_space and pad_pixels >= req_h_space:
            image = unpad(new_img, prev_padding)
        else:
            if image.ndim == 3:
                pad_values = (
                    (req_w_space, req_w_space),
                    (req_h_space, req_h_space),
                    (0, 0),
                )
            elif image.ndim == 2:
                pad_values = ((req_w_space, req_w_space), (req_h_space, req_h_space))
            image = unpad(new_img, pad_values)
        return image

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

    src = np.copy(image)

    src, pad_values = handle_padding(src)

    src_w = src.shape[0]
    src_h = src.shape[1]

    new_img = np.zeros(src.shape, dtype=src.dtype)
    for img_x in range(req_w_space, src_w - req_w_space):
        for img_y in range(req_h_space, src_h - req_h_space):
            v = correlation(src, mask, img_x, img_y)
            new_img[img_x][img_y] = v

    new_img = handle_unpadding(new_img, pad_values)

    return new_img


def make_gradiant(img: np.ndarray, gaussian_size: int):
    img_w = img.shape[0]
    img_h = img.shape[1]
    sigma = gaussian_size / 5
    gaussian = generate_gaussian(sigma, gaussian_size, gaussian_size)
    horizontal_kernel = np.array([[-1, 0, 1]])
    vertical_kernel = np.array([[-1], [0], [1]])
    x_deriv_gaussian = apply_filter(gaussian, horizontal_kernel, 1, 1)
    y_deriv_gaussian = apply_filter(gaussian, vertical_kernel, 1, 1)

    new_img = np.mean(img, axis=2)  # Greyscale
    Mx = apply_filter(new_img, x_deriv_gaussian, math.floor(gaussian_size / 2), 1)
    My = apply_filter(new_img, y_deriv_gaussian, math.floor(gaussian_size / 2), 1)
    gradient = np.zeros((img_w, img_h, 2), dtype=np.float64)
    for img_x in range(img_w):
        for img_y in range(img_h):
            magnitude = math.sqrt(
                math.pow(Mx[img_x][img_y], 2) + math.pow(My[img_x][img_y], 2)
            )
            angle = math.atan2(Mx[img_x][img_y], My[img_x][img_y])
            gradient[img_x][img_y] = [magnitude, angle]
    return gradient


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


def extract_HOG(image: np.ndarray, keypoint):
    WINDOW_SIZE = 16
    BIN_DEG_RANGE = 40
    if 360 % BIN_DEG_RANGE != 0:
        raise ValueError("BIN_DEG_RANGE does not evenly divide 360")

    def deg_to_bin(degree) -> int:
        return math.floor(((degree + 180) % 360) / BIN_DEG_RANGE)

    num_bins = int(360 / BIN_DEG_RANGE)
    offset = math.floor(WINDOW_SIZE / 2)
    histogram = np.zeros(num_bins)
    gradient = make_gradiant(
        safe_slice(
            image,
            keypoint.x - offset,
            keypoint.x + offset + 1,
            keypoint.y - offset,
            keypoint.y + offset + 1,
        ),
        3,
    )

    magnitude = gradient[:, :, 0]
    orientation = gradient[:, :, 1]

    for wx in range(0, WINDOW_SIZE):
        for wy in range(0, WINDOW_SIZE):
            assigned_bin = deg_to_bin(math.degrees(orientation[wx, wy]))
            histogram[assigned_bin] += magnitude[wx, wy]
    return histogram


def feature_matching(image1, image2, detector, extractor):
    NEAREST_MATCH_RATIO_THRESHOLD = 0.5
    if detector != "Moravec" and detector != "Harris":
        raise ValueError("Detector type must be Moravec or Harris")
    if extractor != "LBP" and extractor != "HOG":
        raise ValueError("Extractor Must be LBP or HOG")

    img_1_keypoints = []
    img_2_keypoints = []

    print("Detecting interest points")
    if detector == "Moravec":
        img_1_keypoints = moravec_detector(image1)
        img_2_keypoints = moravec_detector(image2)
    else:
        raise ValueError("Harris detector is not yet implemented")
    img_1_feature_descriptors = []
    img_2_feature_descriptors = []

    display_img(plot_keypoints(image1, img_1_keypoints))
    print("num features 1:", len(img_1_keypoints))
    display_img(plot_keypoints(image2, img_2_keypoints))
    print("num features 2:", len(img_2_keypoints))

    def getAllDescriptors(img, keypoints: List[point], extractorFunc):
        descriptors = []
        for keypoint in keypoints:
            descriptors.append(extractorFunc(img, keypoint))
        return descriptors

    print("Getting all feature descriptors")
    if extractor == "HOG":
        img_1_feature_descriptors = getAllDescriptors(
            image1, img_1_keypoints, extract_HOG
        )
        img_2_feature_descriptors = getAllDescriptors(
            image2, img_2_keypoints, extract_HOG
        )
    else:
        img_1_feature_descriptors = getAllDescriptors(
            image1, img_1_keypoints, extract_LBP
        )
        img_2_feature_descriptors = getAllDescriptors(
            image2, img_2_keypoints, extract_LBP
        )

    def checkTwoMatches(k1, k2):
        feature1 = safe_slice(image1, k1.x - 8, k1.x + 9, k1.y - 8, k1.y + 9)
        feature2 = safe_slice(image2, k2.x - 8, k2.x + 9, k2.y - 8, k2.y + 9)
        display_img(np.concatenate((feature1, feature2), axis=1))

    def matchFeatures(
        img_1_keypoints,
        img_1_feature_descriptors,
        img_2_keypoints,
        img_2_feature_descriptors,
    ):
        img_1_matches = []
        img_2_matches = []
        for i, descriptor1 in enumerate(img_1_feature_descriptors):
            ssd_vals = []
            closest_match: tuple[point] = ()
            for j, descriptor2 in enumerate(img_2_feature_descriptors):
                SSD = np.sum((descriptor1 - descriptor2) ** 2)
                if len(ssd_vals) == 0 or SSD < min(ssd_vals):
                    closest_match = (img_1_keypoints[i], img_2_keypoints[j])
                ssd_vals.append(SSD)

            sorted_ssd_vals = sorted(ssd_vals)
            if sorted_ssd_vals[0] / sorted_ssd_vals[1] < NEAREST_MATCH_RATIO_THRESHOLD:
                img_1_matches.append(closest_match[0])
                img_2_matches.append(closest_match[1])
        return [img_1_matches, img_2_matches]

    print("Matching features")
    return matchFeatures(
        img_1_keypoints,
        img_1_feature_descriptors,
        img_2_keypoints,
        img_2_feature_descriptors,
    )


def plot_matches(image1, image2, matches):
    img_1_with_kp = plot_keypoints(image1, matches[0])
    img_1_matches = matches[0]
    img_2_with_kp = plot_keypoints(image2, matches[1])
    img_2_matches = matches[1]
    img_1_h = image1.shape[1]

    def combine_imgs(img1: np.ndarray, img2: np.ndarray):
        if img1.shape[0] > img2.shape[0]:
            blank = np.zeros((img1.shape[0], img2.shape[1], 3), dtype=img1.dtype)
            blank[0 : img2.shape[0], 0 : img2.shape[1]] = img2
            return np.concatenate((img1, blank), axis=1)
        else:
            blank = np.zeros((img2.shape[0], img1.shape[1], 3), dtype=img1.dtype)
            blank[0 : img1.shape[0], 0 : img1.shape[1]] = img1
            return np.concatenate((blank, img2), axis=1)

    result = combine_imgs(img_1_with_kp, img_2_with_kp)
    for ind in range(0, len(matches[0])):
        p1 = img_1_matches[ind]
        p2 = img_2_matches[ind]
        cv2.line(result, (p1.y, p1.x), (p2.y + img_1_h, p2.x), (0, 0, 255))
    return result


def main():
    img_names = [
        "eye.png",
        "clipped-trees.png",
        "low-contrast-forest.png",
        "low-contrast-rose.png",
        "pretty-tree.png",
        "sandwhich.png",
        "square.jpg",
        "diamond-pattern.jpg",
        "shapes.jpg",
        "pika1.jpg",
        "pika2.jpg",
        "church1.jpg",
        "church2.jpg",
    ]
    img1 = load_img("images(greyscale)/" + img_names[11])
    img1 = aspect_scale(img1, 300)
    img2 = load_img("images(greyscale)/" + img_names[12])
    img2 = aspect_scale(img2, 300)

    display_img(img1)
    display_img(img2)
    matches = feature_matching(img1, img2, "Moravec", "HOG")
    display_img(aspect_scale(plot_matches(img1, img2, matches), 1000))

    # points = moravec_detector(img1)
    # img_w_keypoints = plot_keypoints(img1, points)
    # display_img(aspect_scale(img_w_keypoints,600))
    # test_vals = np.uint8(np.array(
    #     [
    #         [0,0,0,98,0,100],
    #         [0,0,112,145,0,80],
    #         [134,0,0,149,0,70],
    #         [16,0,0,122,0,60],
    #         [16,0,18,0,20,50],
    #         [12,0,12,0,100,70]
    #     ]
    # ))
    # display_img(test_vals)

    # suppressed = non_maxima_suppression(test_vals,90)


# entry point
if __name__ == "__main__":
    main()
