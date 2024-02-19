from cgi import test
from logging.config import valid_ident
import cv2
import numpy as np
import math

# takes a file name and returns an image of type numpy.ndarray
def load_img(file_name):
    return cv2.imread(file_name)

def display_img(image):
    cv2.imshow("eye", image)
    k = cv2.waitKey(0)
    cv2.destroyAllWindows()


def displaySmall(image, window_name: str = ""):
    scaled = cv2.resize(image, (500, 500), interpolation=cv2.INTER_NEAREST)
    cv2.imshow(window_name, scaled)
    k = cv2.waitKey(0)
    cv2.destroyAllWindows()


def gaussianFunc2D(x,y,sigma):
    base = 1 / (2 *math.pi*sigma*sigma)
    exponent = -1*((x*x+y*y)/(2*sigma*sigma))
    result = base * math.exp(exponent)
    # print("2D gaussian for x="+str(x)+" y="+str(y)+" sigma="+str(sigma)+" result: "+str(result))
    return result

def gaussianFunc1D(x,sigma):
    base = 1 / (math.sqrt(2 *math.pi)*sigma)
    exponent = -1*((x*x)/(2*sigma*sigma))
    result = base * math.exp(exponent)
    # print("1D gaussian for x="+str(x)+" sigma="+str(sigma)+" result: "+str(result))
    return result

def generate_gaussian(sigma, filter_w, filter_h):
    if filter_w < 1 or filter_h < 1:
        raise ValueError("filter height and width must be 1 or greater")
    mask = np.zeros((filter_w, filter_h))
    for x in range(0, filter_w):
        for y in range(0,filter_h):
            xVal=x-(filter_w/2)+0.5
            yVal=y-(filter_h/2)+0.5
            v = gaussianFunc2D(xVal,yVal,sigma)
            mask[x][y] = v
    # print("after making mask:\n", mask)
    return mask

def apply_filter(image: np.ndarray, filter: np.ndarray, pad_pixels: int, pad_value: int):        
    def correlation(image: np.ndarray, mask: np.ndarray, img_x: int, img_y: int):
        mask_w =  mask.shape[0]
        mask_h = mask.shape[1]
        
        if mask_w % 2 ==1 and mask_h % 2 == 1: # odd mask size
            val = 0
            for mask_x in range(mask_w):
                for mask_y in range(mask_h):
                    x_diff = int(mask_x-(mask_w/2)+0.5)
                    y_diff = int(mask_y-(mask_h/2)+0.5)
                    pixel =  image[img_x + x_diff][img_y + y_diff]
                    val += pixel[0] * mask[mask_x][mask_y]
            return val
        elif mask_w % 2 ==0 and mask_h % 2 == 0: # even mask size
            raise ValueError("Correlation function does not support even mask sizes")
    def handlePadding(image, pad_value, pad_pixels):
        img = image
        pad_values=((pad_pixels,pad_pixels),(pad_pixels,pad_pixels),(0,0))
        if pad_value == 0:
            img = np.pad(image, pad_values, mode='constant', constant_values=0)
        else:
            img = np.pad(image, pad_values, mode='edge')
        return img
        
    print("before padding:\n", image)
    displaySmall(image,"before Padding")
    img = handlePadding(image, pad_value, pad_pixels)
    print("After padding:\n",img)
    displaySmall(img,"After Padding")
    new_img = np.zeros((img.shape[0],img.shape[1],3), dtype=np.uint8)
    for img_x in range(pad_pixels, img.shape[0]-pad_pixels):
        for img_y in range(pad_pixels, img.shape[1]-pad_pixels):
            v = correlation(img, filter, img_x, img_y)
            print("applying correlation for x:",img_x," y:",img_y," v:",v)
            new_img[img_x][img_y] = v 
    
    displaySmall(new_img,"After Filtering")
    return new_img

# def median_filtering(image, filter_w, filter_h):
# def hist_eq(image):
# def rotate(image, theta):
# def edge_detection(image):

def main():
    imgNames = ['eye.png','clipped-trees.jpg','low-contrast-forest.jpg','low-contrast-rose.jpg','pretty-tree.jpg','sandwhich.png']
    img = load_img('images(greyscale)/' + imgNames[0])
    # display_img(img)
    # k = cv2.waitKey(0) # cloes the window when a key is pressed on the window
    gaussian = generate_gaussian(5,25,25)
    # print("sum gaussian: ", np.sum(gaussian))
    # scaled_image = cv2.resize(gaussian, (500, 500), interpolation=cv2.INTER_NEAREST)
    # display_img(scaled_image)
    # k = cv2.waitKey(0) # cloes the window when a key is pressed on the window
    # checkered_array = np.zeros((4, 4, 3), dtype=np.uint8)
    # checkered_array[1::2, ::2] = [255, 255, 255]
    # checkered_array[::2, 1::2] =  [255, 255, 255]
    # testDot = np.zeros((4,4,3),dtype=np.uint8)
    # testDot[1:3, 1:3] = [255,255,255]
    # bigTestDot= scaled = cv2.resize(testDot, (500, 500), interpolation=cv2.INTER_NEAREST)

    img = apply_filter(img, gaussian,12,0)

# entry point
main()
