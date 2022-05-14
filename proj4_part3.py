import warnings
import numpy as np
import os
import argparse
from student import (calculate_projection_matrix, compute_camera_center)
from helpers import (evaluate_points, visualize_points, plot3dview)
from proj4_part2 import  (camera_calibration,E,K_dash,K,M)
import cv2
import numpy as np
import os
import glob
from pathlib import Path
from matplotlib import pyplot as plt




#Capture the image of your selected object using your cameras

object_dir = os.path.dirname(__file__) #+ '../data/object/'
object_dir = str(Path(object_dir).parents[0])

#images = glob.glob(object_dir+'*.jpeg')

niza_right = cv2.imread('niza_right.jpeg')
niza_left = cv2.imread('niza_left.jpeg')
#print(np.shape(niza_right))
#

niza_right = cv2.cvtColor(niza_right, cv2.COLOR_BGR2GRAY)
niza_left = cv2.cvtColor(niza_left,cv2.COLOR_BGR2GRAY)

cv2.imshow('object left',niza_right)
cv2.waitKey(0)


#Perform SIFT detection and matching so you can identify a set of matched points in the two images. 

sift = cv2.SIFT_create() 
# find the key points and descriptors with SIFT
kp_right, des_right = sift.detectAndCompute(niza_right,None)
kp_left, des_left = sift.detectAndCompute(niza_left,None)

match = cv2.BFMatcher()
matches = match.knnMatch(des_right, des_left, k=2)

good_pts = []
good_pts_index = []
for m,n in matches:
    if m.distance < 0.8*n.distance:
        good_pts.append(m)
        good_pts_index = m.imgIdx

f = open("good_pts.txt", "a")
f.write(str(good_pts_index))
f.close()
        
        
        
images = [
    cv2.drawKeypoints(niza_left,kp_left,None),
    cv2.drawKeypoints(niza_right,kp_right,None)
]
titles = [
    'Left Image with Features',
    'Right Image with Features'
]



def show_images(images, titles):
    #This function is used to show image(s) with titles by sending an array of images and an array of associated titles.
    # images[0] will be drawn with the title titles[0] if exists
    # You aren't required to understand this function, use it as-is.
    assert len(images) == len(titles)
    images = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in images]
    fig = plt.figure()
    n_ims = len(images)
    n = 1
    for image,title in zip(images,titles):
        a = fig.add_subplot(1,n_ims,n)
        if image.ndim == 2: 
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
        n += 1
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_ims)
    plt.show() 
show_images(images, titles)

#checking the epipolar constraint using the Essential matrix computed in PART II. 



#Reconstruct the matched points.
objpoints=cv2.triangulatePoints(M,imgpoints)



M_r,objpoints_r, imgpoints_r,_=camera_calibration(images_dir_r)
M_l,objpoints_l, imgpoints_l,_=camera_calibration(images_dir_l)

S=cv2.stereoCalibrate(objpoints_r, imgpoints_r,imgpoints_l)    # objpoints_r may cause errors ?
E=S[:-2]
'''
# assuming K don't change between objects and chessboard images
K_dash_inv=inv(K_dash)
K_inv=inv(K)
p= left image point of object
p_dash= right image point of object



p_hat=np.dot(K_inv,p)
p_hat_dash=np.dot(K_dash_inv,p_dash)

assert reduce(np.dot, [p_hat.T, E, p_hat_dash]) <0.1 : print("epipolar constranit= ",reduce(np.dot, [p_hat.T, E, p_hat_dash]))


#Reconstruct the matched points.
objpoints=cv2.triangulatePoints(M,imgpoints)


#Draw the reconstructed points in the 3D coordinates and discuss the reconstruction accuracy. 

''' 
'''
like this
draw_params = dict(matchColor = (255,0,0), # draw matches in green color
                   singlePointColor = None,
                   flags = 2)
matching_image = cv2.drawMatches(right_rgb, kp_right, left_rgb, kp_left, good, None, **draw_params)
show_images([matching_image], ['original_image_drawMatches.jpg'])
'''
