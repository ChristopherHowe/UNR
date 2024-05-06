import cv2

# import PIL
# import matplotlib
# import skimage
import numpy as np
import math
import sklearn
import project3 as p3

# # Extract Vocab
# vocab = p3.generate_vocabulary("train_data.txt")

# # Train Object Classifier
# classifier = p3.train_classifier("train_data.txt", vocab)

# # Test Object Classifier

# with open("test_data.txt", "r") as file:
#     for line in file:
#         print(f"extracting features from vocabulary on line {line}")
#         line_parts = line.split()
#         img_name = line_parts[0]
#         test_img = p3.load_img(img_name)
#         out = p3.classify_image(classifier, test_img, vocab)
#         print(f"image: {img_name}, result: {out}")


# Segment an Image
img = p3.load_img("./test-data/test_img.jpg")
im1, im2, im3 = p3.segment_image(img)
p3.display_img(im1)
p3.display_img(im2)
p3.display_img(im3)
