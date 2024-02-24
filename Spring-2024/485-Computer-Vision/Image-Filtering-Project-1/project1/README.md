# Packages Used
I used the following packages in this project: `cv2`, `numpy`, `math`. I chose these over other options since I have some experience with the `numpy` package from my 482 AI class. `cv2` also seemed like a no-brainer since most computer vision tutorials and articles use it, and they have pretty good documentation.

# Gaussian Generation Implementation
The `generate_gaussian` function takes input of a sigma value and a size and returns a NumPy array representing a filter that can be used for blurring. The `generate_gaussian` function uses a helper function `gaussian_func_2d`, which takes any x, y value, and a sigma and returns the resulting height based on the definition of the Gaussian/standard deviation function. The `generate_gaussian` function iterates over every pixel in the blank array and fills it with a value from this helper function.

# Rotation Function
The `rotate` function takes an image and an angle. In order to perform the rotation, the function iterates over all the pixels in the resulting image instead of the source. This prevents any artifacts from the discrete nature of pixels. The `rotate` function makes use of a `rotate_point` helper function that determines the resulting point from any rotation. This can be used to find the source point for a given point in the resulting image by inverting the angle. Each point also needs to be checked whether or not it appears in the original image since the image is not circular. This is done by the helper function `point_in_img`.

# Edge Detection Function
The `edge_detection` function performs four main functions. It uses the Canny edge detector method to determine where the edges in an image are.

First, it copies the image so no changes are made to the source. Then, it applies a small Gaussian mask to make sure that noise is not a factor. This is done with the `apply_filter` function, which iterates over all the pixels in the original image summing the weights of the masks multiplied by their corresponding pixel.

Next, the `edge_detection` function calls the `make_gradient` function. This takes in the original image and applies the first derivative of the Gaussian mask in both the x direction and the y direction to produce `Mx` and `My`. These new images can be squared, added together, and have the square root taken of them to produce the magnitude of the gradient at any point in the image. They can also be partnered with the inverse tangent to produce the normal of the gradient at every point in the image. The magnitude and angle are combined into a new NumPy array to represent the gradient.

Next, the `edge_detection` function calls the `non_maxima_suppression` function. This is responsible for thinning the edges. It does this by using the gradient to determine if a vertex is the 'peak' of an edge or just a part of it. Only peaks are kept, creating thin localized lines.

Finally, the `edge_detection` function uses hysteresis thresholding to remove edges that are not as strong that aren't near other edges. It accomplishes this with a high threshold and a low threshold. It iterates over every pixel, and if a pixel is less than the high threshold but greater than the low threshold, it is included if it is near another pixel that is higher than the high threshold.
