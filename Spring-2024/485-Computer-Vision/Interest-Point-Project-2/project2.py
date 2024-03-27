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
    normalized_image = np.uint8((image - np.min(image)) / (np.max(image) - np.min(image)) * 255)
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
            new_img = np.pad(image, pad_values, mode='constant', constant_values=0)
        else:
            new_img = np.pad(image, pad_values, mode='edge')
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
    TILE_SIZE=15
    TOO_CLOSE_RADIUS=TILE_SIZE/2
    NUM_IPOINTS_PER_TILE=TILE_SIZE


    # Found the idea for this implementation here https://www.ipol.im/pub/art/2018/229/article_lr.pdf
    def suppress_tile_non_maxima(tile: np.ndarray) -> np.ndarray:
        
        def distanceFromPoint(p1, p2):
            return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
        
        def too_close_to_others(p1, others, r) -> bool:
            for other in others:
                if distanceFromPoint(p1, other) < r:
                    return True
            return False

        tile_sorted_inds = np.argsort(tile.flatten())
        sortedInds=[]
        for ind in tile_sorted_inds:
            pos = np.unravel_index(ind, tile.shape)
            sortedInds.append(pos)
        filtered_sorted_inds=[]
        for ind in reversed(sortedInds):
            if tile[ind[0]][ind[1]] != 0:
                filtered_sorted_inds.append(ind)
            else:
                break
        accepted_corners=[]
        resulting_tile=np.zeros(tile.shape, dtype=tile.dtype)
        for ind in filtered_sorted_inds:
            if too_close_to_others(ind, accepted_corners, TOO_CLOSE_RADIUS):
                continue
            accepted_corners.append(ind)
            resulting_tile[ind[0]][ind[1]]=tile[ind[0]][ind[1]]
            if len(accepted_corners) >= NUM_IPOINTS_PER_TILE:
                break
        return resulting_tile

    src_w=src.shape[0]
    src_h=src.shape[1]
    
    # Check that src has intensity data (not rgb)
    if src.ndim != 2:
        raise ValueError("src does not have two dimensions, make sure you are not passing it intensity data")
    # tiles can be any size less than the tile size (edges can be less if corners size is not evenly divided by tile size)
    result=np.zeros(src.shape, dtype=src.dtype)
    
    def get_num_tiles(length, tile_size) -> int:
        if length % tile_size == 0:
            return int(length/tile_size)
        else:
            return math.floor(src_w / tile_size) + 1

    num_tiles_wide=get_num_tiles(src_w, TILE_SIZE)
    num_tiles_tall=get_num_tiles(src_h, TILE_SIZE)

    for tile_x in range(0,num_tiles_wide):
        tile_x_start=tile_x * TILE_SIZE
        tile_x_end = min((tile_x + 1) * TILE_SIZE, src_w)
        for tile_y in range(0,num_tiles_tall):
            tile_y_start=tile_y * TILE_SIZE
            tile_y_end = min((tile_y + 1) * TILE_SIZE, src_h)
            src_tile=src[tile_x_start:tile_x_end, tile_y_start:tile_y_end]
            result_tile = suppress_tile_non_maxima(src_tile)
            result[tile_x_start:tile_x_end, tile_y_start:tile_y_end] = result_tile
            
    return result
       
def rgb2gray(rgb: np.ndarray):
    grey = np.zeros(rgb.shape[:-1], dtype=np.uint8)
    rgb_w = rgb.shape[0]
    rgb_h = rgb.shape[1]
    for rgb_x in range(0,rgb_w):
        for rgb_y in range(0,rgb_h):
            grey[rgb_x][rgb_y] = int(sum(rgb[rgb_x][rgb_y])/3)
    return grey

def binary_img_to_point_arr(image: np.ndarray) -> List[point]:
    points: List[point] = []
    for x in range(0,image.shape[0]):
        for y in range(0,image.shape[1]):
            if image[x][y] == 1:
                points.append(point(x,y))
    return points

# Currently assumes all channels are equal (grey scale)
def moravec_detector(image: np.ndarray) -> List[point]:
    WINDOW_SIZE:int = 3
    THRESHOLD=700
    PRINC_DIRS = [point(0,-1), point(0,1), point(-1,0), point(-1,-1), point(-1,1), point(1,0), point(1,-1), point(1,1)]

    local_src = np.copy(image)
    if image.ndim == 3:
        local_src = rgb2gray(image)

    img_w = local_src.shape[0]
    img_h = local_src.shape[1]

    padding_size: int = math.floor(WINDOW_SIZE/2) + 1  # plus val should be equal to princ dir len
    padded_src, padVals = pad_img(local_src, padding_size, 0)

    def getSW(image: np.ndarray, x: int, y: int, princDir: point):
        # print("x",x,"y",y)
        # print(image[x-1:x+2, y-1:y+2])
        sumDiffs = 0
        for i in range(WINDOW_SIZE):
            window_x:int = x + i - math.floor(WINDOW_SIZE/2)
            for j in range(WINDOW_SIZE):
                window_y=y + j - math.floor(WINDOW_SIZE/2)
                anchorPx = int(image[window_x][window_y])
                shiftedPx = int(image[window_x - princDir.x][window_y - princDir.y])
                pixel_diff= anchorPx - shiftedPx
                sumDiffs += pow(pixel_diff, 2)
        return sumDiffs
                  
    new_img=np.zeros(padded_src.shape, dtype=np.int32)
    for img_x in range(padding_size, img_w - padding_size):
        for img_y in range(padding_size, img_h - padding_size):
            minVal=100000
            for princDir in PRINC_DIRS:
                Sw= getSW(padded_src, img_x, img_y, princDir)
                # print("SW",Sw)
                minVal = min(minVal,Sw)
            # print("min", minVal)
            new_img[img_x][img_y] = minVal
    
    unpadded_img = unpad_img(new_img, padVals)
    suppressed_img = non_maxima_suppression(unpadded_img)
    thresheld_img = threshold(suppressed_img,THRESHOLD)
    return binary_img_to_point_arr(thresheld_img)
    
    

# def harris_detector(image):

# Expects image to be in color
def plot_keypoints(image, keypoints: List[point]):
    def markPx(image,x,y):
        MARK_SIZE=5
        pattern = [
            [0,1,1,1,0],
            [1,0,0,0,1],
            [1,0,1,0,1],
            [1,0,0,0,1],
            [0,1,1,1,0]
        ]
        for lx in range(0,MARK_SIZE):
            for ly in range(0,MARK_SIZE):
                if pattern[lx][ly] == 1:
                    image[x+lx-2][y+ly-2]=[0,0,255]

    result = np.copy(image)
    for keypoint in keypoints:
        markPx(result, keypoint.x, keypoint.y)
    return result

def extract_LBP(image, keypoint: point):
    def get_binary_pattern_as_base_10(image,x,y):
        threshold = image[x][y]
        val=0
        for wx in range(-1,2):
            for wy in range(-1,2):
                if wx == 0 and wy ==0:
                    continue
                val = val * 2 + (1 if image[x+wx][y+wy] > threshold else 0)
        return val
    
    local_src = np.copy(image)
    if image.ndim == 3:
        local_src = rgb2gray(image)

    histogram=np.zeros(256)
    WINDOW_SIZE=16
    img_w=local_src.shape[0]
    img_h=local_src.shape[1]
    for wx in range(0,WINDOW_SIZE):
        img_x=keypoint.x+wx-int(WINDOW_SIZE/2)-1
        for wy in range(0,WINDOW_SIZE):
            img_y=keypoint.y+wy-int(WINDOW_SIZE/2)-1
            if (img_x not in range(1,img_w-1)) or (img_y not in range(1,img_h-1)):
                histogram[0]+=1
            else:
                binary_pattern=get_binary_pattern_as_base_10(local_src, img_x, img_y)
                if binary_pattern not in range(0,256):
                    raise ValueError("Something went wrong extracting binary pattern, value is not 0-255")
                histogram[binary_pattern] += 1
    return histogram





# def extract_HOG(image, keypoint):
# def feature_matching(image1, image2, detector, extractor):
# def plot_matches(image1, image2, matches):

def main():
    img_names = ['eye.png', 'clipped-trees.png', 'low-contrast-forest.png', 'low-contrast-rose.png','pretty-tree.png', 'sandwhich.png','square.jpg','diamond-pattern.jpg','shapes.jpg']
    img = load_img('images(greyscale)/' + img_names[0])
    img = aspect_scale(img,300)
    display_img(img)
    
    LBP = extract_LBP(img, point(100,100))
    print(LBP)
    # points = moravec_detector(img)
    # img_w_keypoints = plot_keypoints(img, points)
    # display_img(aspect_scale(img_w_keypoints,600))
    # display_img_normalized(interest_point_img)
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
