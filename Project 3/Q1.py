import numpy as np
from scipy.linalg import rq

#expressing points as a list
world_coord = np.array([ [0, 0, 0], [0, 3, 0],[0, 7, 0],[0, 11, 0], [7, 1, 0],[0, 11, 7],[7, 9, 0],[0, 1, 7]])

image_coord = np.array([[757, 213],[758, 415],[758, 686],[759, 966],[1190, 172],[329, 1041],[1204, 850],[340, 159]])

#normalising coordinates 
world_coord = (world_coord - np.mean(world_coord, axis=0)) / np.std(world_coord)
image_coord=(image_coord - np.mean(image_coord, axis=0)) / np.std(image_coord)

#Homogenous coordinates 
world_coord=np.hstack((world_coord,(np.ones((world_coord.shape[0],1)))))
image_coord=np.hstack((image_coord,np.ones((image_coord.shape[0],1))))

#finding A matrix 
A=np.empty((0,12))
for i in range(len(world_coord)):
    A1=np.concatenate((np.zeros(4),
                    -world_coord[i].T,
                    (image_coord[i][1]*world_coord[i]).T
                    ))
    A2=np.concatenate((world_coord[i].T,
                    np.zeros(4),
                        (-image_coord[i][0]*world_coord[i]).T,
                    ))
    A3=np.concatenate(((-image_coord[i][1]*world_coord[i]).T,
                    (image_coord[i][0]*world_coord[i]).T,
                    np.zeros(4)
                    ))

    A=np.vstack((A,A1,A2,A3))

print(A.shape)

#finding P
U,S,V= np.linalg.svd(A)
V=V.T
p=V[:,-1]
p=np.reshape(p,(3,4))
unit_elemnt=p[-1,-1]
p=p/unit_elemnt
print(p.shape)
#finding M 
M=p[:,:3]
print(M.shape)
#RQ decompstion 
R,Q= rq(M)
print(R.shape,Q.shape)
one_element=R[-1,-1]
K=R/one_element
rotation=Q
#Finding C 
U,S,V=np.linalg.svd(p)
c=V[:,-1]
print(c.shape)
one_element=c[-1]
c=c/one_element
T=-c

#Printing output and error 
print("\n P \n",p)
print("\n K \n",K)
print("\n R \n",rotation)
print("\n T \n",T)

mean_error=[]
for i in range(len(world_coord)):
    error=np.abs(world_coord[i]-p*world_coord[i])
    error=np.mean(error)
    mean_error.append(error)


print("The Error per Point is",*mean_error,sep="\n")

