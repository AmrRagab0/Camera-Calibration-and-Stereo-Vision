#  Load images and prepare them 
# img_r = cv2.imread('../data/pic_a.jpg')
# img_b = cv2.imread('../data/pic_b.jpg')



# Extract 2d points and map them to 3d point 

# Points_2D = np.loadtxt('../data/pts2d-norm-pic_a.txt')
# Points_3D = np.loadtxt('../data/pts3d-norm.txt')

##                              ????????????????????????????????????

#Calculate the projection matrix given corresponding 2D and 3D points

'''
M_l = calculate_projection_matrix(Points_2D_l,Points_3D_l)
M_r = calculate_projection_matrix(Points_2D_r,Points_3D_r)


# evaluate points
[Projected_2D_Pts_l, Residual_l] = evaluate_points( M_l, Points_2D_l, Points_3D_l)
[Projected_2D_Pts_r, Residual_r] = evaluate_points( M_r, Points_2D_r, Points_3D_r)

'''
####################################################################################

import cv2
import numpy as np
import os
import glob



def camera_calibration(  images_dir):
    '''
    input: path
    
    output: M ,
    
    
    '''


    # Defining the dimensions of checkerboard
    CHECKERBOARD = (6,9)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Creating vector to store vectors of 3D points for each checkerboard image
    objpoints = []
    # Creating vector to store vectors of 2D points for each checkerboard image
    imgpoints = [] 


    # Defining the world coordinates for 3D points
    objp = np.zeros((1, CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
    objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    prev_img_shape = None

    # Extracting path of individual image stored in a given directory
    # images = glob.glob('./images/*.jpg')
    images = glob.glob(images_dir+'*.jpeg')

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        # Find the chess board corners
        # If desired number of corners are found in the image then ret = true
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH+
            cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)

        """
        If desired number of corner are detected,
        we refine the pixel coordinates and display 
        them on the images of checker board
        """
        if ret == True:
            objpoints.append(objp)
            # refining pixel coordinates for given 2d points.
            corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)

            imgpoints.append(corners2)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2,ret)

        cv2.imshow('img',img)
        cv2.waitKey(0)

    cv2.destroyAllWindows()

    h,w = img.shape[:2]

    """
    Performing camera calibration by 
    passing the value of known 3D points (objpoints)
    and corresponding pixel coordinates of the 
    detected corners (imgpoints)
    """
    ret, M, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

    return M,rvecs,tvecs
# evaluate points
# [Projected_2D_Pts_r, Residual_r] = evaluate_points( M, imgpoints, objpoints)

#[Projected_2D_Pts_l, Residual_l] = evaluate_points( M_l, Points_2D_l, Points_3D_l)
#[Projected_2D_Pts_r, Residual_r] = evaluate_points( M_r, Points_2D_r, Points_3D_r)


# print("Camera matrix : \n")
# print(mtx)
# print("dist : \n")
# print(dist)
# print("rvecs : \n")
# print(rvecs)
# print("tvecs : \n")
# print(tvecs)


#compute the fundamental matrix

# F=
# F, mask = cv2.findFundamentalMat(img1_points, img2_points, cv2.FM_RANSAC)
# F=cv2.stereoCalibrate(objpoints, imgpoints_r,imgpoints_l)

# visualize points
# visualize_points(Points_2D_l,Projected_2D_Pts_l)
# visualize_points(Points_2D_r,Projected_2D_Pts_r)


#Calculate the camera center using the M found from previous step
# Center_l = compute_camera_center(M_l)
# Center_r = compute_camera_center(M_r)


