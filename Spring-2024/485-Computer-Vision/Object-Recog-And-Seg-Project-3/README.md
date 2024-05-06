# Readme

## Explain what package you used for image manipulation

I used the cv2, the math and the sklearn.linear_model packages to construct this project. I cv2 was used to handle all the image manipulation tasks such as displaying images. sklearn was a new tool for me to learn a bit more about neural networks. The package makes it easy to set up linear learning models and have somewhat of an understanding of what is going on under the hood.

### `generate_vocabulary(train_data_file: str)`

This function is used to generate a visual vocabulary for a bag-of-words approach. It iterates over a set of training images specified in train_data_file, extracts SIFT (Scale-Invariant Feature Transform) descriptors from each image, and constructs a vocabulary by clustering these descriptors using techniques like K-means clustering. SIFT descriptors capture distinctive local features from images, which are then used for image classification.

### `extract_features(image: np.ndarray, vocabulary: np.ndarray)`

Given an input image and a pre-computed visual vocabulary, this function computes a bag-of-words representation for the image. It extracts SIFT descriptors from the image and then assigns each descriptor to the nearest cluster (word) in the vocabulary. The resulting bag-of-words vector represents the frequency of each word in the vocabulary within the image.

### `train_classifier(train_data_file: str, vocab: np.ndarray)`

This function trains a classifier using a Perceptron model. It reads the training data from train_data_file, extracts bag-of-words features for each image using the extract_features function, and trains a Perceptron classifier on these features. The classifier learns to predict the class labels of images based on their bag-of-words representations.

### `classify_image(classifier: Perceptron, test_img: np.ndarray, vocabulary: np.ndarray)`

Given a trained classifier, an input test image, and the visual vocabulary, this function predicts the class label of the test image. It first extracts the bag-of-words features for the test image using the extract_features function, then feeds these features into the trained classifier to obtain the predicted class label.

### `threshold_image(image, low_thresh, high_thresh)`

This function performs image thresholding, a technique used to separate objects or regions of interest from the background in an image. It converts the input image to grayscale, then applies a threshold to classify pixels as either foreground or background based on their intensity values. This is often used as a preprocessing step in segmentation algorithms.

### `grow_regions(image: np.ndarray)`

Region growing is a segmentation technique that groups pixels into regions based on certain similarity criteria. This function implements region growing by starting from seed points (identified using histogram analysis) and iteratively adding neighboring pixels to the regions if they meet similarity conditions. The regions are iteratively grown until convergence, forming distinct segments in the image.

### `split_regions(image: np.ndarray)`

This function recursively splits regions in an image until they become uniform. It divides the image into smaller blocks and checks if each block is uniform in color. If not, it further splits the block into smaller sub-blocks and repeats the process until each sub-block is uniform. This hierarchical splitting process helps to capture finer details in the image.

### `merge_regions(image: np.ndarray)`

Region merging is a technique used to merge adjacent regions in an image that are similar in color or texture. This function implements region merging by first identifying regions and constructing a Region Adjacency Graph (RAG) based on the similarity between neighboring regions. It then iteratively merges regions with similar characteristics until no further merging is possible, resulting in a segmented image with fewer, larger regions.

### `segment_image(image)`

This function combines various segmentation techniques (thresholding, region growing, and region merging) to segment an input image into meaningful regions. It first applies thresholding to separate foreground from background, then performs region growing to group pixels into regions. Finally, it applies region merging to merge adjacent regions with similar characteristics, resulting in a segmented image with well-defined objects or regions of interest.
