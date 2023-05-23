import cv2 
import numpy as np 
import glob
from matplotlib import pyplot as plt
from copy import deepcopy

path1=r"F:/artroom/*.png"
path2=r"F:/chess/*.png"
path3=r"F:/ladder/*.png"
dataset1= [pointer for pointer in glob.glob(path1)]
dataset2=[pointer for pointer in glob.glob(path2)]
dataset3=[pointer for pointer in glob.glob(path3)]

#creating SIFT AND BF objects 
global sift
sift = cv2.SIFT_create()
global bf 
bf = cv2.BFMatcher()

def match_and_sort_images(pic1,pic2):
    global sift,bf 

    img1=cv2.cvtColor(pic1, cv2.COLOR_BGR2GRAY)
    img2=cv2.cvtColor(pic2, cv2.COLOR_BGR2GRAY)

    #computing features 
    key_points1, descr1 = sift.detectAndCompute(img1,None)
    key_points2, descr2 = sift.detectAndCompute(img2,None)
    img_1=cv2.drawKeypoints(pic1,key_points1,None)

    # Getting matches using BF 
    matches = bf.match(descr1,descr2)
    img_2=cv2.drawMatches(pic1,key_points1,pic2,key_points2,matches,None)
   
    # plt.imshow(img_2)
    # plt.show()
    # Sorting lines based on distance as least distance preferred 
    matches = sorted(matches, key=lambda x:x.distance)
    best_matches=matches[0:100]
    source_points = []
    final_points = []
    count = 1

    #Taking points from within the points obtained 
    for match in best_matches:
        
            p1 = key_points1[match.queryIdx].pt
            p2 = key_points2[match.trainIdx].pt
            source_points.append((p1))
            final_points .append((p2))
            count += 1
           
        
    return source_points,final_points


def normalize_points(points1,points2):
    center_1=np.mean(points1,axis=0)
    center_2=np.mean(points2,axis=0)
    pts1_centered=points1-center_1
    pts2_centered=points2-center_2
    scale_1=np.sqrt(2)/((np.mean(pts1_centered[:,0]**2+pts1_centered[:,1]**2))**0.5)
    scale_2=np.sqrt(2)/((np.mean(pts2_centered[:,0]**2+pts2_centered[:,1]**2))**0.5)
    # scale_1=np.sqrt(2/(np.mean(np.sum((points1-center_1)**2,axis=1))))
    # scale_2=np.sqrt(2/(np.mean(np.sum((points2-center_2)**2,axis=1))))'
    t1_s=np.diag([scale_1,scale_1,1])
    t1_t=np.array([[1,0,-center_1[0]],[0,1,-center_1[1]],[0,0,1]])
    t1=t1_s.dot(t1_t)
    # t1=np.array([[scale_1,0,(-scale_1*center_1[0])],
    #             [0,scale_1,-scale_1*center_1[1]],
    #             [0,0,1]
    #             ])
    t2_s=np.diag([scale_2,scale_2,1])
    t2_t=np.array([[1,0,-center_2[0]],[0,1,-center_2[1]],[0,0,1]])
    t2=t2_s.dot(t2_t)
    # t2=np.array([[scale_2,0,-scale_2*center_2[0]],
    #               [0,scale_2,-scale_2*center_2[1]],
    #               [0,0,1]
    #               ])
    points1=np.concatenate((points1.T,np.ones((1,points1.shape[0]))))
    points2=np.concatenate((points2.T,np.ones((1,points2.shape[0]))))
    nor_points1=np.dot(t1,points1)
    nor_points2=np.dot(t2,points2)
    return nor_points1,nor_points2,t1,t2

def find_f_matrix(points1,points2,t1,t2):
    A=np.zeros((8,9))

    for i in range(A.shape[0]):
        a1=np.array([[points2[:, i][0]*points1[:, i][0],points1[:, i][1]*points2[:, i][0],points2[:, i][0],
        points1[:, i][0]*points2[:, i][1],points1[:, i][1]*points2[:, i][1],points2[:, i][1],
        points1[:, i][0],points1[:, i][1],1 ]])
        # A=np.vstack((A,a1))
        A[i]=a1
    
    # print(A.shape)
    U,S,V= np.linalg.svd(A,full_matrices=True)
    # f=V[-1,:]
    f=V.T[:,-1]
    f=np.reshape(f,(3,3))
    unit_element=f[-1,-1]
    f=f/unit_element
    # print(f)
    U,S,V=np.linalg.svd(f,full_matrices=True)
    S[-1]=0
    f=np.dot(np.dot(U,np.diag(S)),V)
    # unit_element=f[-1,-1]
    # f=f/unit_element
    F=np.dot(t2.T,np.dot(f,t1))
    # unit_element=F[-1,-1]
    # F=F/unit_element
    return F
    # print(F)
    



def ransac_fitting(points1,points2,t1,t2,out_prob=0.5,success_prob=0.99,sample_points=8,thresh=0.5):
    e=out_prob
    p=success_prob
    hypo=[]
    # list_of_points=np.empty(0)
    list_of_points=[]
    inliers=[]
    thresh_result=[]
    samples=1000
    count=0
    max_val=0
    # print(points1.shape[1])
    
    final_inlier=[]
    for i in range(0,samples):
        inliers=[]
        #getting random points 
        points=np.random.randint(0,points1.shape[1],(8,))
        pt_to_check=deepcopy(points)
        pt_to_check=pt_to_check.tolist()
        list_of_points=np.append(list_of_points,np.array([points]))
        pt1=points1[:,points]
        pt2=points2[:,points]
        pt1,pt2=np.array(pt1),np.array(pt2)
        f= find_f_matrix(pt1,pt2,t1,t2)
        for idx in range(points1.shape[1]):
            p1=points1[:,idx].T
            p2=points2[:,idx].T
            error=abs(np.dot(p1.T,np.dot(f,p2)) )
            if error <thresh:
                count=count+1
                inliers.append(idx)
        if len(inliers)>len(final_inlier):
            max_val=count
            count=0
            final_f=f
            final_inlier=inliers
    return final_f,final_inlier


def find_e(k,f):
    e=k.T.dot(f).dot(k)
    U,S,V = np.linalg.svd(e)
    S = [1,1,0]
    e= np.dot(U,np.dot(np.diag(S),V))
    return e
def find_r_t(e,f,k):
    U,S,V=np.linalg.svd(e)
    T1=U[:,2]
    T2=-U[:,2]
    w=np.array([[0,-1,0],
               [1,0,0],
               [0,0,1]])
    R1=np.dot(U,np.dot(w,V))
    R2=np.dot(U,np.dot(w.T,V))
    R_T=[[T1,R1],[T1,R2],[T2,R1],[T2,R2]]
    for i in R_T:
        if np.linalg.det(i[1])<0:
            # print("Updated Val")
            i[0]=-i[0]
            i[1]=-i[1]
    return R_T

def find_best_r_t(R_T,pt1,pt2,k):
    all_3d_points=[]
    for i in R_T:
        C=i[0]
        R=i[1]
        C=np.reshape(C,(3,1))
        # p1=pt1[:-1]
        p1=pt1.T
        # p2=pt2[:-1]
        p2=pt2.T
        projection_matrx_1=np.dot(k, np.dot(np.identity(3), np.hstack((np.identity(3), -1*np.zeros((3, 1))))))
        projection_matrx_2=np.dot(k, np.dot(R, np.hstack((np.identity(3), -1*C))))
        points_3_D=cv2.triangulatePoints(projection_matrx_1,projection_matrx_2,p1,p2)
        points_3_D=points_3_D[0:3]#removing homogenous componenet
        all_3d_points.append(points_3_D)
    count=0
    log=[]
    for i in range(len(all_3d_points)):
        R=R_T[i][1]
        C=R_T[i][0]
        # print(np.linalg.det(R))
        R3=R[2].reshape(1,-1)
        points=all_3d_points[i]
        points=points.T
        # print(i)
        for idx in range(len(points)):
             pt=points[idx]
             val=np.dot(R3,(pt- C))
            #  print(val,pt[2])
             if pt[2]>0:
                 count=count+1
                #  print(count)
        log.append(count)    
        
    max_val=log.index(max(log)) 
    # print(log)
    return R_T[max_val]   

def find_h(img1,img2,p1,p2,f,image_shape,raw_img1=None,raw_img2=None):
    # print(image_shape)
    _,h1,h2 = cv2.stereoRectifyUncalibrated(np.float32(p1), np.float32(p2), f, imgSize=image_shape)        
    img1_rec=cv2.warpPerspective(img1,h1,image_shape)
    img2_rec=cv2.warpPerspective(img2,h2,image_shape)
    rec_img=np.concatenate((img1_rec,img2_rec),axis=1)
    plt.imshow(rec_img)
    plt.show()
    raw_img1=cv2.warpPerspective(raw_img1,h1,image_shape)
    raw_img2=cv2.warpPerspective(raw_img2,h1,image_shape)
    return h1,h2,raw_img1,raw_img2

def drawlines(img1,img2,lines,pts1,pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r,c,_ = img1.shape
    # img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    # img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    pts1=pts1.astype(np.int32)
    pts2=pts2.astype(np.int32)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv2.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv2.circle(img2,tuple(pt2),5,color,-1)
    return img1,img2




def compute_disparity(rect_img1,rect_img2,ndisp,win):
    rect_img1 = cv2.cvtColor(rect_img1, cv2.COLOR_BGR2GRAY)
    rect_img2 = cv2.cvtColor(rect_img2, cv2.COLOR_BGR2GRAY)
    
    rect_img_1_resized, rect_img_2_resized = cv2.resize(rect_img1, (int(rect_img1.shape[1] / 4), int(rect_img1.shape[0] / 4))),cv2.resize(rect_img2, (int(rect_img2.shape[1] / 4), int(rect_img2.shape[0] / 4)))
    
    rect_img_1_resized = rect_img_1_resized.astype(int)
    rect_img_2_resized = rect_img_2_resized.astype(int)

    h, w = rect_img_1_resized.shape
    map = np.zeros((h, w))

    x_diff = w - (2 * win)
    
   
    for y in range(win, h-win):
        
        rect_img_1_blk = []
        rect_img_2_blk = []
        for x in range(win, w-win):
            blk_1 = rect_img_1_resized[y:y + win, x:x + win]
            rect_img_1_blk.append(blk_1.flatten())

            blk_2 = rect_img_2_resized[y:y + win, x:x + win]
            rect_img_2_blk.append(blk_2.flatten())

        rect_img_1_blk = np.array(rect_img_1_blk)
        rect_img_2_blk = np.array(rect_img_2_blk)
        
        rect_img_1_blk = np.repeat(rect_img_1_blk[:, :, np.newaxis], x_diff, axis=2)
        rect_img_2_blk = np.repeat(rect_img_2_blk[:, :, np.newaxis], x_diff, axis=2)
        

        rect_img_2_blk = rect_img_2_blk.T
        
       
        absolute_difference = np.abs(rect_img_1_blk - rect_img_2_blk)
        SAD = np.sum(absolute_difference, axis = 1)
        idx = np.argmin(SAD, axis = 0)
        dis = np.abs(idx - np.linspace(0, x_diff, x_diff, dtype=int)).reshape(1, x_diff)
        map[y, 0:x_diff] = dis 



    map_int = np.uint8(map * 255 / np.max(map))
    plt.imshow(map_int, cmap='hot', interpolation='nearest')
    plt.show()
    plt.imshow(map_int, cmap='gray', interpolation='nearest')
    plt.show()
    return map_int

def compute_depth(map, base, focal_length):
    depth_map = (base * focal_length) / (map + 1e-5)
    depth_map[depth_map > 50000] = 50000
    
    depth_map = np.uint8(depth_map * 255 / np.max(depth_map))

    plt.imshow(depth_map, cmap='hot', interpolation='nearest')
    plt.show()
    plt.imshow(depth_map, cmap='gray', interpolation='nearest')
    plt.show()
    return depth_map




baselinea=536.62
baselinec=97.99
baselinel=228.38
fa=(1733.74+1733.34)/2
fc=(1758.3+1758.23)/2
fl=(1734.16+1734.16)/2
dataset1=[cv2.imread(i) for i in dataset1 ]
points_a1,points_a2=match_and_sort_images(dataset1[0],dataset1[1])
points_a1=np.array(points_a1)
points_a2=np.array(points_a2)
points_a1n,points_a2n,ta1,ta2 =normalize_points(points_a1,points_a2)
# find_f_matrix(points_a1n,points_a2n,ta1,t2a)

f=ransac_fitting(points_a1n,points_a2n,ta1,ta2,thresh=0.5)
# print(np.linalg.det(f[0]))


ka1=np.array([[1733.74, 0, 792.27],
        [0, 1733.74, 541.89],
        [0, 0, 1]])

e=find_e(ka1,f[0])
print("F Matrix for Artroom",f[0])
print("E Matrix for Artroom",e)
R_T=find_r_t(e,f[0],ka1)

pta1_best_n=points_a1n[:,f[1]]
pta2_best_n=points_a2n[:,f[1]]
pta1_best=points_a1[f[1]]
pta2_best=points_a2[f[1]]
best_R_T=find_best_r_t(R_T,pta1_best,pta2_best,ka1)
print("The C nad R are", best_R_T)
width,height,_=dataset1[0].shape



#code taken from https://docs.opencv.org/3.4/da/de9/tutorial_py_epipolar_geometry.html
pt1h=pta1_best[:-1]
pt2h=pta2_best[:-1]
img2=dataset1[1]
img1=dataset1[0]
lines1 = cv2.computeCorrespondEpilines(pt1h.reshape(-1,1,2), 2,f[0])
lines1 = lines1.reshape(-1,3)
img5,img6 = drawlines(img1,img2,lines1,pt1h,pt1h)

# Find epilines corresponding to points in left image (first image) and
# drawing its lines on right image
lines2 = cv2.computeCorrespondEpilines(pt1h.reshape(-1,1,2), 1,f[0])
lines2 = lines2.reshape(-1,3)
img3,img4 = drawlines(img2,img1,lines2,pt2h,pt1h)
plt.subplot(121),plt.imshow(img5)
plt.subplot(122),plt.imshow(img3)
plt.show()

h1,h2,imga1,imga2=find_h(img5,img3,pt1h,pt2h,f[0],(width,height),img1,img2)
ndispa=170
ndispc=220
ndispl=110
deptha1=compute_disparity(imga1,imga2,ndispa,3)
deptha2=compute_depth(deptha1,baselinea,fa)

##############dataset 2 

kc1=np.array([[1758.23, 0, 829.15],
        [ 0, 1758.23, 552.78],
        [ 0 ,0 ,1]])


dataset2=[cv2.imread(i) for i in dataset2 ]
points_c1,points_c2=match_and_sort_images(dataset2[0],dataset2[1])
points_c1=np.array(points_c1)
points_c2=np.array(points_c2)
points_c1n,points_c2n,tc1,tc2 =normalize_points(points_c1,points_c2)
# find_f_matrix(points_a1n,points_a2n,ta1,t2a)

f=ransac_fitting(points_c1n,points_c2n,tc1,tc2,thresh=0.05)
# print(f[0])
# print(np.linalg.det(f[0]))

e=find_e(kc1,f[0])
print("F Matrix for Chess",f[0])
print("E Matrix for Chess",e)


R_T=find_r_t(e,f[0],kc1)

ptc1_best_n=points_c1n[:,f[1]]
ptc2_best_n=points_c2n[:,f[1]]
ptc1_best=points_c1[f[1]]
ptc2_best=points_c2[f[1]]
best_R_T=find_best_r_t(R_T,ptc1_best,ptc2_best,kc1)
print("The C nad R  are", best_R_T)
width,height,_=dataset2[0].shape


#code taken from https://docs.opencv.org/3.4/da/de9/tutorial_py_epipolar_geometry.html
pt1h=ptc1_best[:-1]
pt2h=ptc2_best[:-1]
img2=dataset2[1]
img1=dataset2[0]
lines1 = cv2.computeCorrespondEpilines(pt1h.reshape(-1,1,2), 2,f[0])
lines1 = lines1.reshape(-1,3)
img5,img6 = drawlines(img1,img2,lines1,pt2h,pt1h)

# Find epilines corresponding to points in left image (first image) and
# drawing its lines on right image
lines2 = cv2.computeCorrespondEpilines(pt1h.reshape(-1,1,2), 1,f[0])
lines2 = lines2.reshape(-1,3)
img3,img4 = drawlines(img2,img1,lines2,pt2h,pt1h)
plt.subplot(121),plt.imshow(img5)
plt.subplot(122),plt.imshow(img3)
plt.show()





h1,h2,imgc1,imgc2=find_h(img5,img3,pt1h,pt2h,f[0],(width,height),img1,img2)
depthc1=compute_disparity(imgc1,imgc2,ndispc,3)
depthc2=compute_depth(depthc1,baselinec,fc)


############ dataset 3 
kl1=np.array([[1734.16, 0, 333.49],
        [ 0, 1734.16, 958.05],
        [ 0, 0, 1]])

dataset3=[cv2.imread(i) for i in dataset3 ]
points_l1,points_l2=match_and_sort_images(dataset3[0],dataset3[1])
points_l1=np.array(points_l1)
points_l2=np.array(points_l2)
points_l1n,points_l2n,tl1,tl2 =normalize_points(points_l1,points_l2)

f=ransac_fitting(points_l1n,points_l2n,tl1,tl2,thresh=0.2)
# print(f[0])
# print(np.linalg.det(f[0]))




e=find_e(kl1,f[0])
print("F Matrix for Ladder",f[0])
print("E Matrix for Ladder",e)
R_T=find_r_t(e,f[0],kl1)

ptl1_best_n=points_l1n[:,f[1]]
ptl2_best_n=points_l2n[:,f[1]]
ptl1_best=points_l1[f[1]]
ptl2_best=points_l2[f[1]]
best_R_T=find_best_r_t(R_T,ptl1_best,ptl2_best,kl1)
print("The C nad R are", best_R_T)
width,height,_=dataset2[0].shape


#code taken from https://docs.opencv.org/3.4/da/de9/tutorial_py_epipolar_geometry.html
pt1h=ptl1_best[:-1]
pt2h=ptl2_best[:-1]
img2=dataset3[1]
img1=dataset3[0]
lines1 = cv2.computeCorrespondEpilines(pt1h.reshape(-1,1,2), 2,f[0])
lines1 = lines1.reshape(-1,3)
img5,img6 = drawlines(img1,img2,lines1,pt1h,pt1h)

# Find epilines corresponding to points in left image (first image) and
# drawing its lines on right image
lines2 = cv2.computeCorrespondEpilines(pt1h.reshape(-1,1,2), 1,f[0])
lines2 = lines2.reshape(-1,3)
img3,img4 = drawlines(img2,img1,lines2,pt2h,pt1h)
plt.subplot(121),plt.imshow(img5)
plt.subplot(122),plt.imshow(img3)
plt.show()


h1,h2,imgl1,imgl2=find_h(img5,img3,pt1h,pt2h,f[0],(width,height),img1,img2)
depthl1=compute_disparity(imgl1,imgl2,ndispl,7)
depthl2=compute_depth(depthl1,baselinel,fl)


