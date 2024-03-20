import cv2
import numpy as np
import math

def load_img(file_name):
    return cv2.imread(file_name)

def display_img(image):
    cv2.imshow("image filtering project 1", image)
    k = cv2.waitKey(0)
    cv2.destroyAllWindows()

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

# Currently assumes all channels are equal (grey scale)
def moravec_detector(image: np.ndarray):
    window_size:int = 3
    padding_size: int = math.floor(window_size/2) + 1  # plus val should be equal to princ dir len
    paddedOriginal, padVals = pad_img(image, padding_size, 0)
    principalDirections = [point(0,-1), point(0,1), point(-1,0), point(-1,-1), point(-1,1), point(1,0), point(1,-1), point(1,1)]
    
    img_w = image.shape[0]
    img_h = image.shape[1]
    new_img = np.zeros(image.shape, dtype=np.uint8)

    def getSW(image: np.ndarray, x: int, y: int, princDir: point):
        print("x",x,"y",y)
        # print(image[x-1:x+2, y-1:y+2])
        sumDiffs: int = 0
        for i in range(window_size):
            window_x:int = x + i - math.floor(window_size/2)
            for j in range(window_size):
                window_y=y + j - math.floor(window_size/2)
                anchorPx: int = image[window_x][window_y][0]
                shiftedPx: int = image[window_x - princDir.x][window_y - princDir.y][0]
                sumDiffs += pow(anchorPx - shiftedPx, 2)
        return sumDiffs              
    
    for img_x in range(padding_size, img_w - padding_size):
        for img_y in range(padding_size, img_h - padding_size):
            minVal=1000
            for princDir in principalDirections:
                Sw= getSW(paddedOriginal, img_x, img_y, princDir)
                print("SW",Sw)
                # minVal = min(minVal, )
            new_img[img_x][img_y] = minVal
    
    return new_img

# def harris_detector(image):
# def plot_keypoints(image, keypoints):
# def extract_LBP(image, keypoint):
# def extract_HOG(image, keypoint):
# def feature_matching(image1, image2, detector, extractor):
# def plot_matches(image1, image2, matches):

def main():
    img_names = ['eye.png', 'clipped-trees.png', 'low-contrast-forest.png', 'low-contrast-rose.png','pretty-tree.png', 'sandwich.png']
    img = load_img('images(greyscale)/' + img_names[0])
    ratio = img.shape[0] / img.shape[1]
    height = 500
    img = cv2.resize(img, (height, int(height * ratio)))
    display_img(img)
    interest_point_img = moravec_detector(img)
    display_img(interest_point_img)

# entry point
if __name__ == "__main__":
    main()
