#  Load images and prepare them 
img_r = cv2.imread('../data/pic_a.jpg')
img_b = cv2.imread('../data/pic_b.jpg')



# Extract 2d points and map them to 3d point 

Points_2D = np.loadtxt('../data/pts2d-norm-pic_a.txt')
Points_3D = np.loadtxt('../data/pts3d-norm.txt')

##                              ????????????????????????????????????
# Points_2D_r = 
# up_right_3D = 
# Points_3D_r = 
# #Normalize
# Points_2D_r = 
# Points_3D_r = 

# Points_2D_l = 
# up_left_3D = 
# Points_3D_l = 
# #Normalize
# Points_2D_l =
# Points_3D_l =

#Calculate the projection matrix given corresponding 2D and 3D points


M_l = calculate_projection_matrix(Points_2D_l,Points_3D_l)
M_r = calculate_projection_matrix(Points_2D_r,Points_3D_r)


# evaluate points
[Projected_2D_Pts_l, Residual_l] = evaluate_points( M_l, Points_2D_l, Points_3D_l)
[Projected_2D_Pts_r, Residual_r] = evaluate_points( M_r, Points_2D_r, Points_3D_r)



# visualize points
visualize_points(Points_2D_l,Projected_2D_Pts_l)
visualize_points(Points_2D_r,Projected_2D_Pts_r)


#Calculate the camera center using the M found from previous step
Center_l = compute_camera_center(M_l)
Center_r = compute_camera_center(M_r)


#compute the fundamental matrix

# F=
# F, mask = cv2.findFundamentalMat(img1_points, img2_points, cv2.FM_RANSAC)


