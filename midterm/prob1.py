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