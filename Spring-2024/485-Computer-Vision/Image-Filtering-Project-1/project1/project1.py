import cv2
import numpy as np
import math
import statistics
import matplotlib.pyplot as plt


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
        
    displaySmall(image,"Before Filtering")

    img = handlePadding(image, pad_value, pad_pixels)
    new_img = np.zeros((img.shape[0],img.shape[1],3), dtype=np.uint8)

    for img_x in range(pad_pixels, img.shape[0]-pad_pixels):
        for img_y in range(pad_pixels, img.shape[1]-pad_pixels):
            v = correlation(img, filter, img_x, img_y)
            print("applying correlation for x:",img_x," y:",img_y," v:",v)
            new_img[img_x][img_y] = v 
    
    displaySmall(new_img,"After Filtering")

    return new_img

def median_filtering(image, filter_w, filter_h):
    def getNeighborMedian(image: np.ndarray, filter_w:int, filter_h:int, img_x: int, img_y: int):
        if filter_w % 2 ==1 and filter_h % 2 == 1: # odd mask size
            neighbors = []
            for mask_x in range(filter_w):
                for mask_y in range(filter_h):
                    x_diff = int(mask_x-(filter_w/2)+0.5)
                    y_diff = int(mask_y-(filter_h/2)+0.5)
                    neighbors.append(image[img_x + x_diff][img_y + y_diff][0])
            return statistics.median(neighbors)
        elif filter_w % 2 ==0 and filter_h % 2 == 0: # even mask size
            raise ValueError("Correlation function does not support even mask sizes")
    
    def handlePadding(image, pad_size_x, pad_size_y):
        img = image
        pad_values = ((pad_size_x,pad_size_x),(pad_size_y,pad_size_y),(0,0))
        img = np.pad(image, pad_values, mode='edge')
        return img
    
    displaySmall(image,"Before Filtering")
    
    pad_size_x=int(filter_w/2)
    pad_size_y=int(filter_h/2)    
    img = handlePadding(image, filter_h, filter_w)
    new_img = np.zeros((img.shape[0],img.shape[1],3), dtype=np.uint8)
    
    for img_x in range(pad_size_x, img.shape[0]-pad_size_x):
        for img_y in range(pad_size_y, img.shape[1]-pad_size_y):
            v = getNeighborMedian(img, filter_w, filter_h, img_x, img_y)
            print("applying correlation for x:",img_x," y:",img_y," median:",v)
            new_img[img_x][img_y] = v 
    
    displaySmall(new_img,"After Filtering")

    return new_img

def hist_eq(image):
    probArr = np.zeros((256))
    img_w = image.shape[0]
    img_h = image.shape[1]
    for img_x in range(img_w):
        for img_y in range(img_h):
            intensity = image[img_x][img_y][0]
            print("adding val: ",intensity," for x:",img_x," y: ", img_y )
            probArr[intensity] += 1
    print("Pre Transform historgram:\n", probArr)
    plt.bar(range(len(probArr)), probArr, edgecolor='black', alpha=0.7)
    plt.show()

    totalPx = image.shape[0] * image.shape[1]
    equalizedMapping = np.zeros((256))
    cdf=0
    for ind in range(len(probArr)):
        cdf += probArr[ind]
        equalizedMapping[ind] = 255 * cdf/totalPx
    new_img = np.zeros((img_w,img_h,3), dtype=np.uint8)
    for img_x in range(img_w):
        for img_y in range(img_h):
            px=image[img_x][img_y]
            v = equalizedMapping[px[0]]
            print("Previous Value:",px[0]," New Value:", v)
            new_img[img_x][img_y] = [v,v,v]
    # Checking histogram
    probArr = np.zeros((256))
    for img_x in range(img_w):
        for img_y in range(img_h):
            intensity = new_img[img_x][img_y][0]
            probArr[intensity] += 1
    plt.bar(range(len(probArr)), probArr, edgecolor='black', alpha=0.7)
    plt.show()
    return new_img

            



        

# def rotate(image, theta):
# def edge_detection(image):

def main():
    imgNames = ['eye.png','clipped-trees.png','low-contrast-forest.png','low-contrast-rose.png','pretty-tree.png','sandwhich.png']
    img = load_img('images(greyscale)/' + imgNames[3])
    img = cv2.resize(img, (500, 250), interpolation=cv2.INTER_NEAREST)
    display_img(img)

    # checkered_array = np.zeros((4, 4, 3), dtype=np.uint8)
    # checkered_array[1::2, ::2] = [255, 255, 255]
    # checkered_array[::2, 1::2] =  [255, 255, 255]
    # testDot = np.zeros((4,4,3),dtype=np.uint8)
    # testDot[1:3, 1:3] = [255,255,255]
    # bigTestDot= scaled = cv2.resize(testDot, (500, 500), interpolation=cv2.INTER_NEAREST)
    
    # gaussian = generate_gaussian(5,25,25)
    # img = apply_filter(img, gaussian,12,0)
    # img = median_filtering(img,21,21)
    img = hist_eq(img)
    display_img(img)


# entry point
main()
