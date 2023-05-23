########## Problem 2 ###############
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time 

path = r".\ball.mov"
video = cv2.VideoCapture(path)
time.sleep(2.0)

#HSV values 
lower = (0, 164, 65)
upper = (179, 255, 255)

while True:

    frame = video.read()
    frame = frame[1] 
    if frame is None:
        break

    # cv2.imshow("frame without mask",frame)
    frame1=frame
    # print(frame.shape)
    
    cv2.rectangle(frame,(0,450),(600,600),(0,0,0),-1)
    cv2.imshow("frame",frame1)

    #noise reduction and blurring
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    
    #masking 
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.erode(mask, None, iterations=3)
    mask = cv2.dilate(mask, None, iterations=3)
    cv2.imshow("mask",mask)

    #hough circles
    detected_circles = cv2.HoughCircles(mask, cv2.HOUGH_GRADIENT_ALT,1,minDist=1.5,param1=400,param2=0.75,minRadius=3,maxRadius=10)

    # this was referred from opencv docs
    if detected_circles is not None:
        detected_circles = np.uint32(np.round(detected_circles))
        for idx in detected_circles[0,:]:
            cv2.circle(frame1,(idx[0],idx[1]),idx[2],(0,255,0),2)
            cv2.imshow("Circles",frame1)


    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
            break

	#for frame by frame checking
    # while key not in [ord('q'), ord('k')]:
    #     key = cv2.waitKey(0)
    #     if key == ord("q"):
    #         break


cv2.destroyAllWindows()

########## PROBLEM 3 ##################
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
        line = detected_lines[idx][0]
        cv2.line(warpped, (line[0], line[1]), (line[2], line[3]), (0,0,255), 5, cv2.LINE_AA)

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
        


        #filtering distance
        if  dist < 80 and dist > 10:
            # print(dist)
            distance.append(dist)
        


#avg of calculated distance
avg_distance = np.mean(distance)

print(avg_distance)

cv2.waitKey(0)


##########PROBLEM 4 ###########################
import cv2
import matplotlib.pyplot as plt
import numpy as np

#defining file path and hsv values 
path = r".\hotairbaloon.jpg"
lower=(0,0,243)
upper=(179,255,255)

#reading image
image=cv2.imread(path)
height,width,depth=image.shape
image=cv2.resize(image,(int(width/4),int(height/4)),interpolation=cv2.INTER_NEAREST)


#Image operations 
gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
blur=cv2.GaussianBlur(gray,(21,21),0)
blur=cv2.medianBlur(blur,9)

#Thresholding 
ret,thresh=cv2.threshold(blur,110,150,cv2.THRESH_BINARY)
cv2.imshow("Thresh",thresh)

#finding contours 
contour,hist=cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
count=0
for curves in contour:
    #finding rectangles from determined curves
    x,y,w,h = cv2.boundingRect(curves)

    #random colors 
    blue=int(np.random.randint(0,255,(1,))[0])
    green=int(np.random.randint(0,255,(1,))[0])
    red=int(np.random.randint(0,255,(1,))[0])

    #checking rectangles
    if np.abs(h-w)<40 and  np.abs(h-w)>2 and w!=2*h:
        cv2.rectangle(image,(x,y),(x+w,y+h),(blue,green,red),2)
        count=count+1
       
cv2.putText(image,str("Total Number of Balloons Are "+str(count)),(40,40),cv2.FONT_HERSHEY_COMPLEX,0.5,(blue,green,red),1)       
cv2.imshow("Outline",image)

#finding intensity histogram of image
# plt.hist(image.ravel(), bins=256, fc='k', ec='k')
# plt.show()

cv2.waitKey(0)