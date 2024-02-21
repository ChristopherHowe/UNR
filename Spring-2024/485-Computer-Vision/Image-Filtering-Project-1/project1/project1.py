import cv2
import numpy as np
import math
import statistics

# takes a file name and returns an image of type numpy.ndarray
def load_img(file_name):
    return cv2.imread(file_name)

def display_img(image):
    cv2.imshow("eye", image)
    k = cv2.waitKey(0)
    cv2.destroyAllWindows()


def display_small(image, window_name: str = ""):
    scaled = cv2.resize(image, (500, 500), interpolation=cv2.INTER_NEAREST)
    cv2.imshow(window_name, scaled)
    k = cv2.waitKey(0)
    cv2.destroyAllWindows()

def gaussian_func_2d(x, y, sigma):
    base = 1 / (2 * math.pi * sigma * sigma)
    exponent = -1 * ((x * x + y * y) / (2 * sigma * sigma))
    result = base * math.exp(exponent)
    return result

def generate_gaussian(sigma, filter_w, filter_h):
    if filter_w < 1 or filter_h < 1:
        raise ValueError("filter height and width must be 1 or greater")
    mask = np.zeros((filter_w, filter_h))
    for x in range(0, filter_w):
        for y in range(0, filter_h):
            x_val = x - (filter_w / 2) + 0.5
            y_val = y - (filter_h / 2) + 0.5
            v = gaussian_func_2d(x_val, y_val, sigma)
            mask[x][y] = v
    return mask

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
            image = np.pad(image, pad_values, mode='constant', constant_values=0)
        else:
            image = np.pad(image, pad_values, mode='edge')
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
                pad_values = ((req_w_space, req_w_space), (req_h_space, req_h_space), (0, 0))
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

    display_small(new_img, "After Filtering")

    new_img = handle_unpadding(new_img, pad_values)

    return new_img

def median_filtering(image, filter_w, filter_h):
    def get_neighbor_median(image: np.ndarray, filter_w: int, filter_h: int, img_x: int, img_y: int):
        if filter_w % 2 == 1 and filter_h % 2 == 1:  # odd mask size
            neighbors = []
            for mask_x in range(filter_w):
                for mask_y in range(filter_h):
                    x_diff = int(mask_x - (filter_w / 2) + 0.5)
                    y_diff = int(mask_y - (filter_h / 2) + 0.5)
                    neighbors.append(image[img_x + x_diff][img_y + y_diff][0])
            return statistics.median(neighbors)
        elif filter_w % 2 == 0 and filter_h % 2 == 0:  # even mask size
            raise ValueError("Correlation function does not support even mask sizes")

    def handle_padding(image, pad_size_x, pad_size_y):
        img = image
        pad_values = ((pad_size_x, pad_size_x), (pad_size_y, pad_size_y), (0, 0))
        img = np.pad(image, pad_values, mode='edge')
        return img

    display_small(image, "Before Filtering")

    pad_size_x = int(filter_w / 2)
    pad_size_y = int(filter_h / 2)
    img = handle_padding(image, filter_h, filter_w)
    new_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

    for img_x in range(pad_size_x, img.shape[0] - pad_size_x):
        for img_y in range(pad_size_y, img.shape[1] - pad_size_y):
            v = get_neighbor_median(img, filter_w, filter_h, img_x, img_y)
            new_img[img_x][img_y] = v

    display_small(new_img, "After Filtering")

    return new_img

def hist_eq(image):
    prob_arr = np.zeros((256))
    img_w = image.shape[0]
    img_h = image.shape[1]
    for img_x in range(img_w):
        for img_y in range(img_h):
            intensity = image[img_x][img_y][0]
            prob_arr[intensity] += 1

    total_px = image.shape[0] * image.shape[1]
    equalized_mapping = np.zeros((256))
    cdf = 0
    for ind in range(len(prob_arr)):
        cdf += prob_arr[ind]
        equalized_mapping[ind] = 255 * cdf / total_px
    new_img = np.zeros((img_w, img_h, 3), dtype=np.uint8)
    for img_x in range(img_w):
        for img_y in range(img_h):
            px = image[img_x][img_y]
            v = equalized_mapping[px[0]]
            new_img[img_x][img_y] = [v, v, v]
    # Checking histogram
    prob_arr = np.zeros((256))
    for img_x in range(img_w):
        for img_y in range(img_h):
            intensity = new_img[img_x][img_y][0]
            prob_arr[intensity] += 1
    return new_img

# rotates around the center of the image
# theta should be in radians
def rotate(image, theta):
    # source: https://danceswithcode.net/engineeringnotes/rotations_in_2d/rotations_in_2d.html
    def rotate_point(x0, y0, xc, yc, cos_theta, sin_theta):
        x1 = math.floor((x0 - xc) * cos_theta - (y0 - yc) * sin_theta + xc)
        y1 = math.floor((x0 - xc) * sin_theta + (y0 - yc) * cos_theta + yc)
        return x1, y1

    def point_in_img(x, y):
        return x < img_w and x >= 0 and y < img_h and y >= 0

    new_img = np.zeros(image.shape, dtype=np.uint8)

    img_w = image.shape[0]
    img_h = image.shape[1]

    cos_theta = math.cos(-1 * theta)
    sin_theta = math.sin(-1 * theta)
    center_x = int(math.floor(img_w / 2 + 0.5) - 1)
    center_y = int(math.floor(img_h / 2 + 0.5) - 1)

    for img_x in range(img_w):
        for img_y in range(img_h):
            src_x, src_y = rotate_point(img_x, img_y, center_x, center_y, cos_theta, sin_theta)
            if point_in_img(src_x, src_y):
                og_px = image[src_x][src_y]
                new_img[img_x][img_y] = og_px

    return new_img

def edge_detection(image):
    img_w = image.shape[0]
    img_h = image.shape[1]

    def get_xy_deriv_gaussian(gaussian_size):
        sigma = gaussian_size / 5
        gaussian = generate_gaussian(sigma, gaussian_size, gaussian_size)
        horizontal_kernel = np.array([[-1, 0, 1]])
        vertical_kernel = np.array([[-1], [0], [1]])
        x_deriv_gaussian = apply_filter(gaussian, horizontal_kernel, 1, 1)
        y_deriv_gaussian = apply_filter(gaussian, vertical_kernel, 1, 1)
        return x_deriv_gaussian, y_deriv_gaussian

    def smooth_and_make_gradient(img: np.ndarray, gaussian_size: int):
        x_deriv_g, y_deriv_g = get_xy_deriv_gaussian(gaussian_size)

        new_img = img.astype(np.int16)
        new_img = np.mean(new_img, axis=2)  # Greyscale

        Mx = apply_filter(new_img, x_deriv_g, math.floor(gaussian_size / 2), 1)
        My = apply_filter(new_img, y_deriv_g, math.floor(gaussian_size / 2), 1)

        gradient = np.zeros((img_w, img_h, 2), dtype=new_img.dtype)
        for img_x in range(img_w):
            for img_y in range(img_h):
                magnitude = math.sqrt(math.pow(Mx[img_x][img_y], 2) + math.pow(My[img_x][img_y], 2))
                # magnitude = Mx[img_x][img_y] + My[img_x][img_y]
                angle = math.atan2(Mx[img_x][img_y], My[img_x][img_y])
                gradient[img_x][img_y] = [magnitude, angle]
        return gradient

    def non_maxima_suppression(gradient: np.ndarray):
        def suppress_if_not_peak(px, v1, v2):
            return 0 if v1 > px or v2 > px else px

        edges = np.copy(gradient[:, :, 0:1])
        edges = np.pad(edges, ((1, 1), (1, 1), (0, 0)), mode='edge')

        for img_x in range(img_w):
            for img_y in range(img_h):
                angle = math.degrees(abs(gradient[img_x][img_y][1]))
                px = edges[img_x][img_y]
                if angle < 22.5 or angle > 157.5:
                    edges[img_x][img_y] = suppress_if_not_peak(px, edges[img_x + 1][img_y], edges[img_x - 1][img_y])
                elif angle < 67.5:
                    edges[img_x][img_y] = suppress_if_not_peak(px, edges[img_x + 1][img_y + 1], edges[img_x - 1][img_y - 1])
                elif angle < 112.5:
                    edges[img_x][img_y] = suppress_if_not_peak(px, edges[img_x][img_y + 1], edges[img_x][img_y - 1])
                else:
                    edges[img_x][img_y] = suppress_if_not_peak(px, edges[img_x + 1][img_y - 1], edges[img_x - 1][img_y + 1])
        return edges

    def hysteresis_threshold(img: np.ndarray, low: int, high: int):
        def threshold(img: np.ndarray, threshold: int):
            new_img = np.where(img < threshold, 0, 1)
            return new_img

        def neighbor_not_zero(img_x, img_y):
            for i in range(-1, 2):
                for j in range(-1, 2):
                    if high_t_img[img_x + i][img_y + j] != 0:
                        return 1
            return 0

        low_t_img = threshold(img, low)
        high_t_img = threshold(img, high)
        result = np.copy(high_t_img)
        for img_x in range(img_w):
            for img_y in range(img_h):
                if low_t_img[img_x][img_y] != 0:
                    result[img_x][img_y] = neighbor_not_zero(img_x, img_y

)
        return result

    new_img = np.copy(image)
    new_img = apply_filter(new_img, generate_gaussian(1, 3, 3), 2, 1)
    gradient = smooth_and_make_gradient(new_img, 3)
    magnitude_map = gradient[:, :, 0:1]
    angle_map = gradient[:, :, 1:]
    edges = non_maxima_suppression(gradient)
    better_edges = hysteresis_threshold(edges, 10, 20)

    return (better_edges * 255).astype(np.uint8)

def main():
    img_names = ['eye.png', 'clipped-trees.png', 'low-contrast-forest.png', 'low-contrast-rose.png',
                 'pretty-tree.png', 'sandwich.png']
    img = load_img('images(greyscale)/' + img_names[0])
    ratio = img.shape[0] / img.shape[1]
    height = 500
    img = cv2.resize(img, (height, int(height * ratio)))
    display_img(img)

    checkered_array = np.zeros((12, 12, 3), dtype=np.uint8)
    checkered_array[1::2, ::2] = [255, 255, 255]
    checkered_array[::2, 1::2] = [255, 255, 255]
    gaussian = generate_gaussian(1, 4, 4)

    test_dot = np.zeros((7, 7, 3), dtype=np.uint8)
    test_dot[2:5, 2:5] = [255, 255, 255]

    # img = edge_detection(img)
    # img = apply_filter(img, gaussian, 0, 0)
    img  = rotate(img, math.pi/4)
    display_img(img)

# entry point
if __name__ == "__main__":
    main()
