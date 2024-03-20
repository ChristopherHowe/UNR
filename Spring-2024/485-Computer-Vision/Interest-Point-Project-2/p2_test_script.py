import cv2
import PIL
import matplotlib
import skimage
import numpy as np
import math
import project2 as p2


#Load and display image
img1 = p2.load_img("test_img1.jpg")
p2.display_img(img)
img2 = p2.load_img("test_img2.jpg")
p2.display_img(img2)

#Extract Keypoints with Moravec Detector
keypoints = p2.moravec_detector(img1) 

#Plot Keypoints
key_img = p2.plot_keypoints(img1, keypoints)
p2.display_img(key_img)

#Perform feature matching with Moravec and LBP
matches = p2.feature_matching(img1,img2,"Moravec", "LBP")
match_img = p2.plot_matches(img1,img2,matches)
p2.display_img(match_img)

#Perform feature matching with Moravec and HOG
matches = p2.feature_matching(img1,img2,"Moravec", "HOG")
match_img = p2.plot_matches(img1,img2,matches)
p2.display_img(match_img)


#Perform feature matching with Harris and LBP
matches = p2.feature_matching(img1,img2,"Harris", "LBP")
match_img = p2.plot_matches(img1,img2,matches)
p2.display_img(match_img)

#Perform feature matching with Harris and HOG
matches = p2.feature_matching(img1,img2,"Harris", "LBP")
match_img = p2.plot_matches(img1,img2,matches)
p2.display_img(match_img)

