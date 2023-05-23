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