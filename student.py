# Projection Matrix Stencil Code
# Written by Eleanor Tursman, based on previous work by Henry Hu,
# Grady Williams, and James Hays for CSCI 1430 @ Brown and
# CS 4495/6476 @ Georgia Tech

import numpy as np
from numpy.linalg import inv


# Returns the projection matrix for a given set of corresponding 2D and
# 3D points. 
# 'Points_2D' is nx2 matrix of 2D coordinate of points on the image
# 'Points_3D' is nx3 matrix of 3D coordinate of points in the world
# 'M' is the 3x4 projection matrix
def calculate_projection_matrix(Points_2D, Points_3D):
    # To solve for the projection matrix. You need to set up a system of
    # equations using the corresponding 2D and 3D points:
    #
    #                                                     [M11       [ u1
    #                                                      M12         v1
    #                                                      M13         .
    #                                                      M14         .
    # [ X1 Y1 Z1 1 0  0  0  0 -u1*X1 -u1*Y1 -u1*Z1          M21         .
    #  0  0  0  0 X1 Y1 Z1 1 -v1*X1 -v1*Y1 -v1*Z1          M22         .
    #  .  .  .  . .  .  .  .    .     .      .          *  M23   =     .
    #  Xn Yn Zn 1 0  0  0  0 -un*Xn -un*Yn -un*Zn          M24         .
    #  0  0  0  0 Xn Yn Zn 1 -vn*Xn -vn*Yn -vn*Zn ]        M31         .
    #                                                      M32         un
    #                                                       M33         vn ]
    #                   World_points                                 Image_points
    #
    #
    # Then you can solve this using least squares with the 'np.linalg.lstsq' operator.
    # Notice you obtain 2 equations for each corresponding 2D and 3D point
    # pair. To solve this, you need at least 6 point pairs. Note that we set
    # M34 = 1 in this scenario. If you instead choose to use SVD via np.linalg.svd, you should
    # not make this assumption and set up your matrices by following the 
    # set of equations on the project page. 
    #
    ##################
    Points_3D=np.array(Points_3D)
    N_of_points = Points_3D.shape[0]
    # init
    Image_points = np.zeros((N_of_points*2,1))
    World_points = np.zeros((N_of_points*2,11)) # as described above

    # setting the matrices 
    for i in range(0,N_of_points,2):
        World_points[i,:] = [Points_3D[i,0],Points_3D[i,1] ,Points_3D[i,2] ,1, 0, 0,0, 0 ,-Points_2D[i,0]*Points_3D[i,0], -Points_2D[i,0]*Points_3D[i,1],-Points_2D[i,1]*Points_3D[i,2]]
        #print(i)
        #World_points[i,:] = [Points_3D[i,0],Points_3D[i,1],Points_3D[i,2],4,5,6,7,8,9,10,11]
        World_points[i+1,:] = [0,0,0,0,Points_3D[i,0],Points_3D[i,1],Points_3D[i,2],1, -Points_2D[i,1]*Points_3D[i,0], -Points_2D[i,1]*Points_3D[i,1], -Points_2D[i,1]*Points_3D[i,2]]
        Image_points[i] = [Points_2D[i,0] ]
        Image_points[i+1] = [Points_2D[i,1]]

    # world_points * M = image points
    # so, M = world points inverse * image points

   
    # A = Worldpoint    B = image points

    A_T_A = np.dot(np.mat(World_points).T,np.mat(World_points))
    A_T_B = np.dot(np.mat(World_points).T,np.mat(Image_points))

    A_T_A_inv = inv(A_T_A)
    M = np.dot(A_T_A_inv,A_T_B)
    M = M.T
    M = np.array(M)
    print(M.shape)

    M = np.append(M,[1])
    M = M.reshape((3,4))

    


    ##################

    # This M matrix came from a call to rand(3,4). It leads to a high residual.
    # Your total residual should be less than 1.
    #print('Randomly setting matrix entries as a placeholder')
    #M = np.array([[0.1768, 0.7018, 0.7948, 0.4613],
     #             [0.6750, 0.3152, 0.1136, 0.0480],
      #            [0.1020, 0.1725, 0.7244, 0.9932]])

    return M


# Returns the camera center matrix for a given projection matrix
# 'M' is the 3x4 projection matrix
# 'Center' is the 1x3 matrix of camera center location in world coordinates
def compute_camera_center(M):
   
    # Center = -Q^-1 * m4 

    Q = M[:,0:3]
    m4 = M[:,3]
    Q_inv = inv(Q)

    Center = -Q_inv*m4


    # Replace this with the correct code
    # In the visualization you will see that this camera location is clearly
    # incorrect, placing it in the center of the room where it would not see all
    # of the points.
    # Center = np.array([1, 1, 1])

    return Center


'''


# Testing Calculate Projection matrix func
Points_3D= np.array([[312.747,309.140, 30.086]
,[305.796, 311.649, 30.356]
,[307.694, 312.358 ,30.418]
,[310.149, 307.186 ,29.298]
,[311.937, 310.105, 29.216]
,[311.202, 307.572, 30.682]
,[307.106, 306.876, 28.660]
,[309.317, 312.490, 30.230]
,[307.435 ,310.151, 29.318]
,[308.253 ,306.300, 28.881]
,[306.650 ,309.301, 28.905]
,[308.069 ,306.831, 29.189]
,[309.671 ,308.834, 29.029]
,[308.255 ,309.955, 29.267]
,[307.546 ,308.613, 28.963]
,[311.036 ,309.206, 28.913]
,[307.518 ,308.175, 29.069]
,[309.950 ,311.262, 29.990]
,[312.160 ,310.772, 29.080]
,[311.988, 312.709, 30.514]]
)

Points_2D = np.array([[880,  214],
 [43,  203],
[270,  197],
[886,  347],
[745,  302],
[943 , 128],
[476 , 590],
[419 , 214],
[317 , 335],
[783 , 521],
[235 , 427],
[665 , 429],
[655 , 362],
[427 , 333],
[412 , 415],
[746 , 351],
[434 , 415],
[525 , 234],
[716 , 308],
[602 , 187]])

#print(Points_2D)
#print(Points_3D)

M = calculate_projection_matrix(Points_2D,Points_3D)
print(M)

'''
