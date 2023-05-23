import cv2
import matplotlib.pyplot as plt
import numpy as np


path = r".\train_track.jpg"

lower=(0,0,245)
upper=(179,255,255)

distance = []   

#resiziing image
image=cv2.imread(path)
image=cv2.resize(image,(640,480),interpolation=cv2.INTER_NEAREST)
# cv2.imshow("image",image)
# plt.imshow(image)
# plt.show()


#definig ROI and Final points
ROI=np.float32([[296,276],[198,480],[448,480],[343,276]])
# desireed_view=np.float32([[160,120],[160,240],[240,240],[240,120]])
desireed_view=np.float32([[160,120],[160,480],[240,480],[240,120]])
# cv2.fillPoly(image,np.int32([ROI]),(255,255,255),)
# cv2.imshow("image",image)

#getting perspective matrix 
matrix=cv2.getPerspectiveTransform(ROI,desireed_view)
# print(matrix)

#warping image
warpped= cv2.warpPerspective(image,matrix,(480,640), flags=cv2.INTER_LINEAR)
cv2.imshow("Warpped",warpped)

#HSV conversion
hsv = cv2.cvtColor(warpped, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv, lower, upper)
cv2.imshow("mask",mask)

#detecting lines using HoughLinesP
#code referred from opencv Docs
detected_lines=cv2.HoughLinesP(mask,1,np.pi/180,50,None,50,10)
if detected_lines is not None:
    for idx in range(0, len(detected_lines)):
        l = detected_lines[idx][0]
        cv2.line(warpped, (l[0], l[1]), (l[2], l[3]), (0,0,255), 5, cv2.LINE_AA)

cv2.imshow("Lines",warpped)

#calculating distance between lines 
            
for idx1 in range(len(detected_lines)):   
    for idx2 in range(idx1+1, len(detected_lines)):
        x1, y1, x2, y2 = detected_lines[idx1][0]
        x3, y3, x4, y4 = detected_lines[idx2][0]
        
        #finding midpoints and distance between midpoints
        mid1 = (x1 + x2) // 2
        mid2 = (x3 + x4) // 2
        dist = abs(mid1 - mid2)
        
        #print(dist)

        #filtering distance
        if  dist < 80 and dist > 10:
            print(dist)
            distance.append(dist)
        


#avg of calculated distance
avg_distance = np.mean(distance)

print(avg_distance)

cv2.waitKey(0)