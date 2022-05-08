#  Load images and prepare them



#Extract 2d points and map them to 3d points




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
