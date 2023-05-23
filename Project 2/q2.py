import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
from copy import deepcopy
from scipy.spatial.transform import Rotation as R
import imageio
import imutils



path1=r".\image_1.jpg"
path2=r".\image_2.jpg"
path3=r".\image_3.jpg"
path4=r".\image_4.jpg"
image1 = cv2.imread(path1)
image2 = cv2.imread(path2)
image3 = cv2.imread(path3)
image4 = cv2.imread(path4)


image1=cv2.resize(image1,(int(image1.shape[0]/4),int(image1.shape[1]/4)))
image2=cv2.resize(image2,(int(image2.shape[0]/4),int(image2.shape[1]/4)))
image3=cv2.resize(image3,(int(image3.shape[0]/4),int(image3.shape[1]/4)))
image4=cv2.resize(image4,(int(image4.shape[0]/4),int(image4.shape[1]/4)))

#creating SIFT AND BF objects 
global sift
sift = cv2.SIFT_create()
global bf 
bf = cv2.BFMatcher(cv2.NORM_L1,crossCheck = True)

#finding homography matrix 
def find_homography(pixel_points,world_points):
    A=[]
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
    U,S,V=np.linalg.svd(A)
    V=V.T
    h=V[:,-1] #last col
    # h=V[-1,:]#last row
    h=h.reshape((3,3))
    # print(h[2][2])
    value=h[2][2]
    h=h/value
    return h

#Finding the inlier and outlier points for each hypothesis of RANSAC
def eval_hypo(points_1,points_2,h_matrix,threshold=0.5):
    distance=np.empty(1,)
    for i in range(points_1.shape[0]):
        pt1=points_1[i]
        pt1=np.append(pt1,[1])
        pt2=points_2[i]
        pt2=np.append(pt2,[1])
        calculated_points=np.dot(h_matrix,pt1)
        value=calculated_points[2]
        calculated_points=calculated_points/value
        dist=calculated_points-pt2
        dist=np.linalg.norm(dist)**2
        
        #Log of all distances of Points 
        distance=np.append(distance,dist)

    #Number of inliers
    success=np.where(distance<=threshold)[0].shape[0]
    # print("Pos. points",success)
    return success

#RANSAC fitting 
def ransac_fitting(points_1,points_2,out_prob=0.5,success_prob=0.99,sample_points=4):
    e=out_prob
    p=success_prob
    hypo=[]
    # list_of_points=np.empty(0)
    list_of_points=[]
    inliers=np.empty(0)
    thresh_result=[]
    
    samples=int(np.log(1 - p) / np.log(1 - np.power((1 - e), sample_points)))
    
    #setting number of samples to 2500 since Calulated samples number is too low 
    samples=1000
    for i in range(0,samples):
        #getting random points 
        points=np.random.randint(0,points_1.shape[0],(4,))
        pt_to_check=deepcopy(points)
        pt_to_check=pt_to_check.tolist()
        #ensuring points do not repeat
        if pt_to_check in list_of_points:
            print("Points already present renewing")
            continue
            
        else:
            # print(points)
            # print("Points added")
            list_of_points=np.append(list_of_points,points)
            pt1=points_1[points]
            pt2=points_2[points]
            # print(pt1,pt2)
            #doing fitting
            coef=find_homography(pt2,pt1)
            # print("Computed Homography")
            #finding best hypo
            success=eval_hypo(points_1,points_2,coef)
            # print("Computed Hypo")
            thresh_result.append(success)
            inliers=np.append(inliers,success)
            hypo.append(coef)
            # print("Added homo matrix to list")
            print(i)
            

    
    #Selecting the best Value 
    
    best_hypo=np.argmax(inliers)
  
    return hypo[best_hypo]

def match_and_sort_images(pic1,pic2):
    global sift,bf 

    img1=cv2.cvtColor(pic1, cv2.COLOR_BGR2GRAY)
    img2=cv2.cvtColor(pic2, cv2.COLOR_BGR2GRAY)

    #computing features 
    key_points1, descr1 = sift.detectAndCompute(img1,None)
    key_points2, descr2 = sift.detectAndCompute(img2,None)


    # Getting matches using BF 
    matches = bf.match(descr1,descr2)

    # Sorting lines based on distance as least distance preferred 
    matches = sorted(matches, key=lambda x:x.distance)
    source_points = []
    final_points = []
    count = 1

    #Taking random points from within the points obtained 
    for match in matches:
        
            p1 = key_points1[match.queryIdx].pt
            p2 = key_points2[match.trainIdx].pt
            source_points.append(p1)
            final_points .append(p2)
            count += 1
           
        
    return source_points,final_points,matches,key_points1,key_points2,descr1,descr2

def homography_ransac(start_points,final_points):
    # homography 
    H=ransac_fitting(np.array(final_points),np.array(start_points))
    return H 

def match_and_warp_output(start_img,final_img,matches,keypoints1,keypoints2,H,image_1):
    
    # Drawing mathces on images from keypoints
    img3 = cv2.drawMatches(start_img,keypoints1,final_img,keypoints2,matches[10:40], None)
    plt.imshow(cv2.cvtColor(img3,cv2.COLOR_BGR2RGB))
    plt.title("Feature Match using SIFT")
    plt.show()
    
    # Warping the image 
    final = cv2.warpPerspective(image3, H, ((image_1.shape[1] + final_img.shape[1]), final_img.shape[0])) #wraped image
    plt.imshow(cv2.cvtColor(final,cv2.COLOR_BGR2RGB))
    plt.title("Warped Image")
    plt.show()
    return final


def sitching_images(start_image,final_image,image_2,title):
    # Stitching the images 

    final_image[0:image_2.shape[0], 0:image_2.shape[1]] = image_2 #stitched image
    
    plt.imshow(cv2.cvtColor(final_image,cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.show()
    return final_image


#stitching part by part
img1_pts,img2_pts,match_1_2,kp1,kp2,des1,des2=match_and_sort_images(image1,image2)
H=homography_ransac(img1_pts,img2_pts)
warped_img=match_and_warp_output(image1,image2,match_1_2,kp1,kp2,H,image1)
title="Img 1 and 2 stitched"
result_img_1_2=sitching_images(image2,warped_img,image1,title=title)


img2_pts,img3_pts,match_2_3,kp2,kp3,des2,des3=match_and_sort_images(image2,image3)
H=homography_ransac(img2_pts,img3_pts)
warped_img=match_and_warp_output(image3,image2,match_2_3,kp2,kp3,H,image1)
title="Img 2 and 3 stitched"
result_img_2_3=sitching_images(image3,warped_img,image2,title=title)


img3_pts,img4_pts,match_3_4,kp3,kp4,des3,des4=match_and_sort_images(image3,image4)
H=homography_ransac(img3_pts,img4_pts)
warped_img=match_and_warp_output(image4,image3,match_3_4,kp3,kp4,H,image1)
title="Img 3 and 4 stitched"
result_img_3_4=sitching_images(image4,warped_img,image3,title=title)


#stitching parts of image 
img3_pts,img4_pts,match_3_4,kp3,kp4,des3,des4=match_and_sort_images(result_img_1_2,result_img_2_3)
H=homography_ransac(img3_pts,img4_pts)
warped_img=match_and_warp_output(result_img_2_3,result_img_1_2,match_3_4,kp3,kp4,H,image1)
title="Img 1_2 and 2_3 stitched"
result_img_1_2_3=sitching_images(result_img_2_3,warped_img,result_img_1_2,title=title)

#stitching whole image
img3_pts,img4_pts,match_3_4,kp3,kp4,des3,des4=match_and_sort_images(result_img_3_4,result_img_1_2_3)
H=homography_ransac(img3_pts,img4_pts)
warped_img=match_and_warp_output(result_img_1_2_3,result_img_3_4,match_3_4,kp3,kp4,H,image1)
title="Img 1_2_3 and 3_4 stitched"
result_img_1_2_3=sitching_images(result_img_1_2_3,warped_img,result_img_3_4,title=title)



