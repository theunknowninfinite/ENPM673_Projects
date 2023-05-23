import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
from copy import deepcopy
from scipy.spatial.transform import Rotation as R


# file path 
path=r".\project2.avi"
video = cv2.VideoCapture(path)
time.sleep(2.0)
counter=0

#defining world points and K matrix 
world_coords=np.array([[0,0],
                          [27.9,0],
                          [0,21.9],
                          [27.9,21.6]])

K=np.array([[1.38E+03	,0	,9.46E+02],
            [0,	1.38E+03	,5.27E+02],
            [0	,0	,1]])
#empty for rotation and translation
roll,pitch,yaw=[],[],[]
t_x,t_y,t_z=[],[],[]

def hough_transform(edge_image):
    edges=edge_image
    edge_height,edge_width=edge_image.shape
    rho_length=int(np.round((np.sqrt(np.square(edge_height) + np.square(edge_width))),0))
    # print(rho_length)
    #rho ranges from - daigonal to + diagonal and theta from zero to 180
    accumulator=np.zeros((2*rho_length,180))
    # print(accumulator.shape)
    #theta values from 0 to 180 with a step of one
    theta=np.arange(0,180,1)
    cos_t=np.cos(np.deg2rad(theta))
    sin_t=np.sin(np.deg2rad(theta))
    x,y=np.nonzero(edges>0)
    for idx in range(len(x)):
            x_point=x[idx]
            y_point=y[idx]
            rho=x_point*cos_t+y_point*sin_t
            rho=rho.astype(np.int32)
            for index in range(len(theta)):
                accumulator[rho[index],theta[index]] = accumulator[rho[index],theta[index]]+1
    #getting top 6 max values 
    number = 6
    idx = np.argsort(accumulator.ravel())[-number:][::-1]
    max_values = accumulator.ravel()[idx]
    lines= np.c_[np.unravel_index(idx, accumulator.shape)]
    # print(lines)
    return lines 

def merge_lines(lines):
    line_5 = deepcopy(lines)
    final_lines=np.zeros((2,))
    #Sort lines that close to each other
    while True:
        values=[]
        values_t=[]
        if len(line_5)>0:
            line1=line_5[-1]
            line_5=np.delete(line_5,-1,0)
            line_5=line_5[0:]
            for i in range(len(line_5)): 
                value_to_minus=lines[i]
                rho_diff=np.abs(line1[0]-value_to_minus[0])
                theta_diff=np.abs(line1[1]-value_to_minus[1])
                values.append(rho_diff)
                values_t.append(theta_diff)
            values=np.array(values)
            values_t=np.array(values_t)
            if  np.any(values<30):
                flag=True
                # print("Line rejected")
            else:
                final_lines=np.append(final_lines, line1,0)
                flag=False
                # print("Line Added")                    
        else:
            break
    #final lines of hough transform without similar lines 
    final_lines=final_lines[2:]
    final_lines=final_lines.reshape((4,2))
    return final_lines

def find_intersection_points(line1,line2):
        #finding corners of paper 
    rho1, theta1 = line1
    rho2, theta2 = line2
    if np.abs(theta1-theta2)<89:
        #  print("parallel Lines")
        #parallel lines to be ignored 
         return None
    theta = np.array([
        [np.cos(np.deg2rad(theta1)), np.sin(np.deg2rad(theta1))],
        [np.cos(np.deg2rad(theta2)), np.sin(np.deg2rad(theta2))]
    ])
    rho = np.array([[rho1], [rho2]])
    x, y = np.linalg.solve(theta, rho)
    x, y = int(np.round(x)), int(np.round(y))

    return [x,y]

def draw_hough_lines(lines,img):
    corner_points=[]
    for point in lines:
        rho,theta=point
        x1=-100
        y1=np.int64((rho-(x1*np.cos(np.deg2rad(theta))))/(np.sin(np.deg2rad(theta))))
        x2=3000
        y2=np.int64((rho-(x2*np.cos(np.deg2rad(theta))))/(np.sin(np.deg2rad(theta))))
        cv2.line(img,(y1,x1),(y2,x2) ,(0,0,255), 3, cv2.LINE_AA)
        cv2.imshow("Lines",img)
    for idx,idx1 in ((0,1),(0,2),(0,3),(1,2),(1,3),(2,3)):
        points=find_intersection_points(lines[idx],lines[idx1])
        if points is not None:
            # print(points)
            corner_points.append(points)
            cv2.circle(img,(int(points[1]),int(points[0])),2,(0,255,0),3)
            #drawing points in the form of y and x but labelled as x and y 
            cv2.putText(img,str((points[1],points[0])),(int(points[1]+10),int(points[0]+10)),fontFace=cv2.FONT_HERSHEY_PLAIN,color=(255,0,0),thickness=2,fontScale=2)
        cv2.imshow("Circles",img)
        
    #in form of x and y  
    return corner_points
def find_origin(points):
    #min value of X 
    index=min(points)
    return index 

def find_homography(pixel_points,world_points):
    #ax=b
    A=[]
    #mking A matrix 
    for i in range(len(pixel_points)): 
        px1,py1= pixel_points[i]
        wx1,wy1=world_points[i]
        arr=np.array([[ wx1 ,wy1, 1, 0, 0,0,-(px1*wx1),-(px1*wy1),-px1 ],
            [0,0,0,wx1,wy1,1,-(py1*wx1),-(py1*wy1),-py1] 
            ])
        A.append(arr)
    A=np.array(A)
    A=A.reshape((8,9))
    # print(A)
    #SVD  for H
    U,S,V=np.linalg.svd(A)
    V=V.T
    #Last Col of V is the H matrix
    h=V[:,-1] #last col
    # h=V[-1,:]#last row
    h=h.reshape((3,3))
    # print(h[2][2])
    value=h[2][2]
    h=h/value
    #measuring true value and calulated points 
    for i in range(len(pixel_points)):
        p_p= pixel_points[i]
        w_p=world_points[i]
        w_p=np.append(w_p,[1])
        test_point=np.matmul(h,w_p)
        p_p=np.append(p_p,[1])
        # print(np.mean(p_p-test_point))        

    return h


def find_rotation_translation(k_matrix,h_matrix):
    #doing kinvH= lamda r1r2T
    k=k_matrix
    h=h_matrix
    #pseudo inverse 
    k_inverse=np.linalg.pinv(k)
    k_h=np.matmul(k_inverse,h)
    lamda_1=np.linalg.norm(k_h[:,0])#first col
    lamda_2=np.linalg.norm(k_h[:,1])#second col
    lamda = np.mean((lamda_1,lamda_2))
    k_h_l=k_h/lamda
    r1=k_h_l[:,0]
    r2=k_h_l[:,1]
    r3=np.cross(r1,r2)
    t=k_h_l[:,2]
    r=np.stack((r1,r2,r3),axis=1)
    #checking if unit vector 
    # print(np.linalg.norm(r1))
    # print(np.linalg.norm(r2))
    # print(np.linalg.norm(r3))
    return r,t

def update_r_t(roll,pitch,yaw,tx,ty,tz,rot,translation):
    #updating R and T 
    r=R.from_matrix(rot)
    angles=r.as_euler('zyx', degrees=True)
    translation_x=translation[0]
    translation_y=translation[1]
    translation_z=translation[2]

    x=angles[2]
    y=angles[1]
    z=angles[0]

    roll.append(x)
    pitch.append(y)
    yaw.append(z)
    tx.append(translation_x)
    ty.append(translation_y)
    tz.append(translation_z)

    return roll,pitch,yaw,tx,ty,tz




while True:
    image = video.read()
    image = image[1] 
    if image is None:
        print(counter)
        break
    counter=counter+1
    gray_img=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    #converting to gray
    # cv2.imshow("Gray",gray_img)
    #blur to remove noise and focus on paper and remove other objects
    gray_img=cv2.GaussianBlur(gray_img, (15,15), 0)
    # cv2.imshow("Gray",gray_img)
    edge_img=cv2.Canny(gray_img,60,160)
    cv2.imshow("Edges",edge_img)
   
    #drawing output of hough lines 
    image1=deepcopy(image)
    line=hough_transform(edge_img)
    final_lines=merge_lines(line)
    corners=draw_hough_lines(final_lines,image1) 

    # print(corners)
    min_point=find_origin(corners)
    corners=np.array(corners)
    min_point=np.array(min_point)

    #sorted based on x values , in image frame, it is y value 
    arranged_corners=corners[corners[:, 0].argsort()]
    new_corners=arranged_corners-min_point

    # picture form(y,x),true points (x,y)
    # print(new_corners)
    #fidning Homography
    H=find_homography(new_corners,world_coords)
    # finding r and T 
    r,t=find_rotation_translation(K,H)
    #udpating values 
    roll,pitch,yaw,t_x,t_y,t_z=update_r_t(roll,pitch,yaw,t_x,t_y,t_z,r,t)
    
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
            break

	#for frame by frame checking
    # while key not in [ord('q'), ord('k')]:
    #     key = cv2.waitKey(0)
    #     if key == ord("q"):
    #         break

#Plotting RPY and XYZ values 
plt.plot(roll,label="Roll")
plt.plot(pitch,label="Pitch")
plt.plot(yaw,label="Yaw")
plt.xlabel("Frame")
plt.ylabel("Angle(Degrees)")
plt.legend()
plt.show()

plt.plot(t_x,label="T X")
plt.plot(t_y,label="T Y")
plt.plot(t_z,label="T Z")
plt.xlabel("Frame")
plt.ylabel("Movement(CM)")
plt.legend()
plt.show()