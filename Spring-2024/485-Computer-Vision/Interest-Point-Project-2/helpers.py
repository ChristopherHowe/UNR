import numpy as np
import cv2
# from project2 import safe_slice, display_img


def aspect_scale(img: np.ndarray, height):
    ratio = img.shape[0] / img.shape[1]
    return cv2.resize(img, (height, int(height * ratio)))


def display_img_normalized(image: np.ndarray):
    # normalized_image = np.zeros(image.shape, dtype=np.uint8)
    # cv2.normalize(image, normalized_image, 0,255,cv2.NORM_MINMAX)
    normalized_image = np.uint8((image - np.min(image)) / (np.max(image) - np.min(image)) * 255)
    cv2.imshow("image filtering project 1", normalized_image)
    k = cv2.waitKey(0)
    cv2.destroyAllWindows()


# def checkTwoMatches(image1, image2, k1, k2):
#     feature1 = safe_slice(image1, k1.x - 8, k1.x + 9, k1.y - 8, k1.y + 9)
#     feature2 = safe_slice(image2, k2.x - 8, k2.x + 9, k2.y - 8, k2.y + 9)
#     display_img(np.concatenate((feature1, feature2), axis=1))
