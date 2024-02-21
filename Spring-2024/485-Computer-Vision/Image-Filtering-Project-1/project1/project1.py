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

def displayFractionPoints(image: np.ndarray, window_name: str = ""):
    new_img = np.copy(image)
    max_intensity = np.max(new_img)
    new_img = new_img / max_intensity * 255
    new_img = np.clip(new_img, 0, 255).astype(np.uint8)
    # print(new_img)
    displaySmall(new_img, window_name)


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

# TODO: add support for filters of even w or even h
# TODO: make sure the func does not change the og image
# TODO: make sure this works for intensities
# apply filter should not require masks to be displayable
def apply_filter(image: np.ndarray, mask: np.ndarray, pad_pixels: int, pad_value: int):        
    def correlation(image: np.ndarray, mask: np.ndarray, img_x: int, img_y: int):
        val:int = 0
        for mask_x in range(mask_w):
            for mask_y in range(mask_h):
                x_diff = int(mask_x-(mask_w/2)+0.5)
                y_diff = int(mask_y-(mask_h/2)+0.5)
                pixel =  image[img_x + x_diff][img_y + y_diff]
                src_val = pixel if np.isscalar(pixel) else pixel[0]
                step = src_val * mask[mask_x][mask_y]
                # print("handling correlation for mask x: ",mask_x,"mask y:",mask_y," img x:",img_x," img y:",img_y, "step:", step)
                val += step
        return val
    
    def handlePadding(image, pad_value, pad_pixels):
        pad_values=()
        if image.ndim == 3:
            pad_values=((pad_pixels,pad_pixels),(pad_pixels,pad_pixels),(0,0))
        else:
            pad_values=((pad_pixels,pad_pixels),(pad_pixels,pad_pixels))

        if pad_value == 0:
            image = np.pad(image, pad_values, mode='constant', constant_values=0)
        else:
            image = np.pad(image, pad_values, mode='edge')
        return image, pad_values
    
    def unpad(x, pad_width):
        # this unpad function comes from this stack overflow question https://stackoverflow.com/a/57956349
        slices = []
        for c in pad_width:
            e = None if c[1] == 0 else -c[1]
            slices.append(slice(c[0], e))
        return x[tuple(slices)]

    
    def handleMaskCheck(mask: np.ndarray):
        if mask.ndim == 1: # make 1D arrays into 2D with width 1
            mask = mask.reshape(1,-1)
        if mask.ndim > 2:
            raise ValueError("Does not support masks with a higher dimension than 2")
        for i in range(mask.ndim):
            if mask.shape[i] % 2 == 0:
                raise ValueError("Correlation function does not support even mask sizes")
            if math.floor(mask.shape[i]/2) > pad_pixels:
                raise ValueError("number of pixels to pad (" + str(pad_pixels) + ") is not substantial enough to handle the mask size(", str(mask.shape[i]) + ")")
        return mask


    # displaySmall(image,"Before Filtering")

    mask = handleMaskCheck(mask)
    mask_w = mask.shape[0]
    mask_h = mask.shape[1] if mask.ndim == 2 else 0

    src =np.copy(image)
    # print("before padding src shape:", src.shape)

    src, pad_values = handlePadding(src, pad_value, pad_pixels)
    # print("after padding src shape:", src.shape)

    src_w = src.shape[0]
    src_h = src.shape[1]
    # displaySmall(image,"After padding")

    
    new_img = np.zeros(src.shape, dtype=src.dtype)
    # print("after creating new image shape:", new_img.shape)
    for img_x in range(pad_pixels, src_w-pad_pixels):
        # print("handling row x=:", img_x, "in image")
        for img_y in range(pad_pixels, src_h-pad_pixels):
            v = correlation(src, mask, img_x, img_y)
            # print("applying correlation for x:", img_x," y:",img_y," v:",v)
            new_img[img_x][img_y] = v 
    
    # displaySmall(new_img,"After Filtering")
    
    new_img = unpad(new_img, pad_values)

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

# rotates around the center of the image
# theta should be in radians
def rotate(image, theta):
    # source: https://danceswithcode.net/engineeringnotes/rotations_in_2d/rotations_in_2d.html
    def rotate_point(x0, y0, xc, yc, cosTheta, sinTheta):
        x1 = math.floor((x0 - xc) * cosTheta - (y0 - yc) * sinTheta + xc)
        y1 = math.floor((x0 - xc) * sinTheta + (y0 - yc) * cosTheta + yc)
        return x1, y1
    
    def pointInImg(x,y):
        return x < img_w and x >=0 and y < img_h and y >= 0

    new_img = np.zeros((img_w, img_h, 3), dtype=np.uint8)

    cosTheta = math.cos(-1 * theta)
    sinTheta = math.sin(-1 * theta)
    
    img_w = image.shape[0]
    img_h = image.shape[1]

    center_x:int = math.floor(img_w / 2 + 0.5) - 1
    center_y:int = math.floor(img_h / 2 + 0.5) - 1
    
    for img_x in range(img_w):
        for img_y in range(img_h):
            src_x, src_y = rotate_point(img_x, img_y, center_x, center_y, cosTheta, sinTheta)
            if pointInImg(src_x,src_y):
                og_px = image[src_x][src_y]
                new_img[img_x][img_y] = og_px

    return new_img

# TODO: fix smoothing so that not reliant on even mask size
def edge_detection(image):
    img_w = image.shape[0]
    img_h = image.shape[1]

    def getXYDerivGaussian(gaussianSize):
        sigma = gaussianSize / 5
        gaussian = generate_gaussian(sigma,gaussianSize,gaussianSize)
        print("Generating Gaussian Derivatives")
        horizontalKernal = np.array([[-1,0,1]])
        verticalKernel = np.array([[-1],[0],[1]])
        x_deriv_guassian = apply_filter(gaussian,horizontalKernal,1,1) 
        y_deriv_guassian = apply_filter(gaussian,verticalKernel,1,1)
        return x_deriv_guassian, y_deriv_guassian        

    def smoothAndMakeGradiant(img:np.ndarray, gaussianSize:int):
        x_deriv_G, y_deriv_G = getXYDerivGaussian(gaussianSize)

        new_img = img.astype(np.int16)
        new_img = np.mean(new_img, axis=2) # Greyscale

        print("Determining Mx")
        Mx = apply_filter(new_img, x_deriv_G,math.floor(gaussianSize/2),1)
        print("Determining My")
        My = apply_filter(new_img, y_deriv_G,math.floor(gaussianSize/2),1)
        
        print("Creating the Gradiant")
        gradiant = np.zeros((img_w, img_h,2), dtype = new_img.dtype)
        for img_x in range(img_w):
            for img_y in range(img_h):
                magnitude = math.sqrt(math.pow(Mx[img_x][img_y],2) + math.pow(My[img_x][img_y],2))
                angle = math.atan2(My[img_x][img_x], My[img_x][img_y])
                gradiant[img_x][img_y]= [magnitude, angle]
        return gradiant

    def nonMaximaSupression(gradiant: np.ndarray):
        def suppressIfNotPeak(px, v1, v2):
            return 0 if v1 > px or v2 > px else px
        
        edges = np.copy(gradiant[:,:,0:1])
        edges = np.pad(edges,((1,1),(1,1),(0,0)), mode='edge')
        
        for img_x in range(img_w):
            for img_y in range(img_h):
                angle = math.degrees(abs(gradiant[img_x][img_y][1]))
                px = edges[img_x][img_y]
                if angle < 22.5 or angle > 157.5:
                    edges[img_x][img_y] = suppressIfNotPeak(px,edges[img_x+1][img_y],edges[img_x-1][img_y])
                elif angle < 67.5:
                    edges[img_x][img_y] = suppressIfNotPeak(px,edges[img_x+1][img_y+1],edges[img_x-1][img_y-1])
                elif angle <112.5:
                    edges[img_x][img_y] = suppressIfNotPeak(px,edges[img_x][img_y+1],edges[img_x][img_y-1])
                else:
                    edges[img_x][img_y] = suppressIfNotPeak(px,edges[img_x+1][img_y-1],edges[img_x-1][img_y+1])
        return edges
                
    def threshold(img: np.ndarray, threshold: int):
        new_img = np.where(img < threshold,  0, 1)
        return new_img
    
    def hysteresisThreshold(img: np.ndarray, low: int, high: int):
        def neighborNotZero(img_x,img_y):
            for i in range(-1,2):
                for j in range(-1,2):
                    if high_t_img[img_x+i][img_y+j] != 0:
                        return 1
            return 0
        
        low_t_img = threshold(img, low)
        high_t_img = threshold(img, high)
        result = np.copy(high_t_img)
        for img_x in range(img_w):
            for img_y in range(img_h):
                if low_t_img[img_x][img_y] != 0:
                    result[img_x][img_y] = neighborNotZero(img_x,img_y)
        return result



    
    # Localization
    
    new_img = np.copy(image)
    gradiant = smoothAndMakeGradiant(new_img,3)
    magnitudeMap = gradiant[:,:,0:1]
    angleMap = gradiant[:,:,1:]
    print(angleMap[:,0:10])
    print("Max angle map: ", np.max(angleMap), "min:", np.min(angleMap))
    displayFractionPoints(magnitudeMap,"Magnitude map")
    edges = nonMaximaSupression(gradiant)
    displayFractionPoints(edges,"Edges")
    print("edges max:", np.max(edges),"min:", np.min(edges))
    betterEdges = hysteresisThreshold(edges, 10,10)
    displayFractionPoints(betterEdges,"Hysterized Edges")
    

    # print("Making thresholded map")
    # new_img = threshold(new_img,10)
    # displayFractionPoints(new_img,"Thresholded")

    # return smoothImg(new_img)
    # new_img = smoothImg(new_img)
    # new_img = applyFirstDeriv(new_img)
    return betterEdges

def main():
    imgNames = ['eye.png','clipped-trees.png','low-contrast-forest.png','low-contrast-rose.png','pretty-tree.png','sandwhich.png']
    img = load_img('images(greyscale)/' + imgNames[5])
    img = cv2.resize(img, (800, 800), interpolation=cv2.INTER_NEAREST)
    display_img(img)

    # checkered_array = np.zeros((4, 4, 3), dtype=np.uint8)
    # checkered_array[1::2, ::2] = [255, 255, 255]
    # checkered_array[::2, 1::2] =  [255, 255, 255]
    # gaussian = generate_gaussian(1,5,5)

    # testDot = np.zeros((7,7,3),dtype=np.uint8)
    # testDot[2:5, 2:5] = [255,255,255]

    # bigTestDot= scaled = cv2.resize(testDot, (500, 500), interpolation=cv2.INTER_NEAREST)
    
    # img = apply_filter(img, gaussian,12,0)

    # img = median_filtering(img,21,21)
    # img = hist_eq(img)
    # display_img(img)
    # displaySmall(img)
    # img = rotate(img, math.pi/6)
    # displaySmall(img)
    img = edge_detection(img)
    # display_img(img)
    # displaySmall(img,"after edge")

# entry point
main()
