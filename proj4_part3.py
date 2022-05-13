import warnings
import numpy as np
import os
import argparse
from student import (calculate_projection_matrix, compute_camera_center)
from helpers import (evaluate_points, visualize_points, plot3dview)
import cv2
import numpy as np
import os
import glob




#Capture the image of your selected object using your cameras
object_dir = os.path.dirname(__file__) + '../data/object/'

images = glob.glob(images_dir+'*.jpeg')




#Perform SIFT detection and matching so you can identify a set of matched points in the two images. 

sift = cv2.xfeatures2d.SIFT_create()
# find the key points and descriptors with SIFT
kp_left, des_left = sift.detectAndCompute(left_gray, None)
kp_right, des_right = sift.detectAndCompute(right_gray, None)

match = cv2.BFMatcher()
matches = match.knnMatch(des_right, des_left, k=2)

good = []
for m,n in matches:
    if m.distance < 0.8*n.distance:
        good.append(m)
        
        
        
images = [
    cv2.drawKeypoints(left_rgb,kp_left,None),
    cv2.drawKeypoints(right_rgb,kp_right,None)
]
titles = [
    'Left Image with Features',
    'Right Image with Features'
]
show_images(images, titles)

#checking the epipolar constraint using the Essential matrix computed in PART II. 





#Reconstruct the matched points.
objpoints=cv2.triangulatePoints(M,imgpoints)


#Draw the reconstructed points in the 3D coordinates and discuss the reconstruction accuracy. 

''' like this
draw_params = dict(matchColor = (255,0,0), # draw matches in green color
                   singlePointColor = None,
                   flags = 2)

matching_image = cv2.drawMatches(right_rgb, kp_right, left_rgb, kp_left, good, None, **draw_params)
show_images([matching_image], ['original_image_drawMatches.jpg'])

'''



