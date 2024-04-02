# Packages Used
I used the numpy, math and cv2 packages. I realized quickly during this project that my understanding of everything that you can do with the numpy package is limited and it was good getting to be a little more familiar with it.

# Moravec Detector Implementation
The Moravec detector shifts a window in that 8 principal directions and calculates the Sw value for each window. The corner-ness value of a pixel is equal to the minimum Sw value for each window shift in the principal direction. The Sw value is calculated by the sum of the squared differences between the anchor window (anchor centered on the pixel in question) and the shifted window. Before determining the Sw values, it adds enough padding to account for the size of the window shifted around and the shifting movements. After all the values have been calculated, the Moravec detector removes the padding. It then applies the nonmaximal suppression function, applies the normalizedThreshold function, and returns a list of points representing corners. The list of points is created by the `binary_img_to_point_arr` function which just iterates over the image and appends any nonzero values to the point array.

# Non Maxima Suppression
The corner nonmaxima suppression I implemented is based on a journal article I found on the Harris Detector. The general approach, as I understood it, is to divide the image into tiles. Within each tile, two constraints must be met. First, there can be at max, n corners in a tile. Second, corners cannot be within a certain radius of another corner. Preference is given to corners with higher values.

My implementation of this function first determines how many tiles it will need to split the image into. Then, for each tile of size 15px, it sorts all of the nonzero cornerness values. It then iterates over this sorted list from highest to lowest adding values to an accepted corner array. For each point added, it checks that it’s not too close to any of the other ones. It does this until 15 points have been added to the array.

# Normalized Threshold
The normalized threshold function is the same as a regular thresholding function but instead of keeping pixels that are at or above a certain value, which can change based on how sharp the image is, the normalized threshold function takes a percentage of points to be included as a parameter.

# Harris Detector Implementation
This function implements the Harris corner detection algorithm, which identifies corners or interest points in an image. It begins by converting the input image to grayscale and applying Sobel filters to compute gradients in both the x and y directions. These gradients are then used to calculate the elements of the autocorrelation matrix for each pixel. Next, the function computes the corner response function using the elements of the autocorrelation matrix and a constant parameter (K_VAL). 

It iterates through each pixel and computes the corner response value, determining the R-value for each pixel in the image. A high positive R value corresponds to a corner. Low values (flat regions) are filtered out by the threshold and negative values (edges) and filtered out by the max function. After all the R values have been determined, the same nonmaximal suppression and normalized thresholding functions used in the Moravec detector are used.

# Extract LBP Features Implementation
The LBP feature extractor takes in an image and a key point value and returns a corresponding feature descriptor. It produces a histogram representing the area around the key point. This function creates a 16 by 16 window around the key point. For each pixel in the window, this function makes a binary pattern of the 3 by 3 window around the pixel. This is done by calling the `get_binary_pattern_as_base_10` function. This function iterates over the 3 by 3 window and uses this statement `val = val * 2 + (1 if px > threshold else 0)` to create a 0-255 base 10 value for each 3 by 3 window.

# Extract HOG features Implementation
The Extract_HOG function takes in an image and a key point value and returns a corresponding feature descriptor. It produces a histogram representing the area around the key point. This function works by creating a window of size 16 by 16 and using the gradient to fill the histogram. For each pixel in the window, the orientation of the gradient at that pixel corresponds to the bin in the histogram. The value added to the histogram for a particular pixel is the magnitude of the gradient. This function does not take into account any partial partitioning of the magnitude.

# Feature Matching Implementation
The function matches features by first detecting interest points in both images using either the Moravec or Harris detector. Then, it extracts feature descriptors for each interest point using either the LBP or HOG extractor. These descriptors represent the local image structure around each key point. Next, it compares the descriptors between the two images, finding the closest match for each descriptor pair based on the sum of squared differences (SSD). To ensure accurate matches, it applies a ratio test on the SSD values, filtering out matches below a certain threshold. Finally, it returns the matched key points from both images, providing a reliable set of correspondences between the features of the two images.
