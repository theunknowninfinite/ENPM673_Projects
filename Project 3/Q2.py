import numpy as np
import cv2 as cv
import glob

#code was referred from https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
generated_world_points = np.zeros((9*6,3), np.float32)
generated_world_points[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

#multiplying with value of length of side of square 
generated_world_points=generated_world_points*22

# Arrays to store object points and image points from all the images.
world_points = [] # 3d points
image_points = [] # 2d points

#loading Images 
img = [pointer for pointer in glob.glob(r'.\Calibration_Imgs\\*.jpg')]

for file in img:
    img = cv.imread(file)
    height,width,_=img.shape
    img=cv.resize(img,(int(width/3),int(height/3)))
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (9,6), None)
    # If found, add object points, image points (after refining them)
    if ret == True:
        world_points.append(generated_world_points)
        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        image_points.append(corners2)
        # Draw and display the corners
        cv.drawChessboardCorners(img, (9,6), corners2, ret)
        cv.imshow('img', img)
        cv.waitKey(500)
cv.destroyAllWindows()

#calibrating Camera 
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(world_points, image_points, gray.shape[::-1], None, None)

#finding error 
error_list=[]
for i in range(len(world_points)):
    image_points2, _ = cv.projectPoints(world_points[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv.norm(image_points[i], image_points2, cv.NORM_L2)/len(image_points2)
    error_list.append(error)

#list of all errors and K matrix 
print( "list of errors: ",*error_list,sep="\n")
print("K",mtx)