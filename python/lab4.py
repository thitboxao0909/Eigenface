import os
import numpy as np
import matplotlib.pyplot as plt
import pca
import math
import re
from PIL import Image
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier



def get_imlist(path):
	# Returns a list of filenames for all jpg images in a directory.
 	return [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.pgm')]

train_image_names = get_imlist('yalefaces')

im = np.array(Image.open(train_image_names[0])) # open one image to get size
m, n = im.shape[0:2] # get the size of the images
imnbr = len(train_image_names) # get the number of images

#print(train_image_names)
#print(train_image_names[2])
print( m )
print(n)
print(imnbr)



# create matrix to store all flattened images
immatrix = np.array([np.array(Image.open(im)).flatten() for im in train_image_names], 'f')
print("George",immatrix[1])
 
# perform PCA
eigenvectors, eigenvalues, mean_face = pca.pca(immatrix)

# show some images (mean and 7 first modes)
plt.figure()
plt.gray()
plt.subplot(2,4,1)
plt.title("mean")
plt.imshow(mean_face.reshape(m,n))
plt.axis('off')

# eigenface_titles = ["eigenface %d" % i for i in range(7)]
for i in range(7):
     plt.subplot(2,4,i+2)
     plt.imshow(eigenvectors[i].reshape(m,n))
     plt.title("eigenface %d"%(i+1))
     plt.axis('off')
     
#plt.show()

for k in range(eigenvalues.size):
     R=sum(eigenvalues[:k])/sum(eigenvalues)
     print(eigenvalues[k])
     print('this is R: ', R)

# PART 2 - QUES 2
print ('end R')
threhold = 0.95
for k in range(eigenvalues.size):
     R=sum(eigenvalues[:k])/sum(eigenvalues)
     if R > threhold:
        print('this is K2: ',k)
        break
print ('end R')
threhold = 0.85
for k in range(eigenvalues.size):
     R=sum(eigenvalues[:k])/sum(eigenvalues)
     if R > threhold:
        print('this is K1: ',k)
        break

normalised_training = np.ndarray(shape=(len(train_image_names), m*n))

for i in range(len(train_image_names)):
    normalised_training[i] = np.subtract(immatrix[i],mean_face)

U = np.array(eigenvectors[:k]).transpose()
trainW = np.dot(normalised_training, U)
 
test_path= 'test\George_W_Bush_0061.pgm'
unknown_face = Image.open(test_path)
test_image = np.array(unknown_face, dtype='float64').flatten()

print('george test image',test_image)


normalised_test = np.subtract(test_image, mean_face)
testW = np.dot(normalised_test, U)
print(testW)

testW=testW.reshape(1, testW.shape[0]) # ensure the weights of an image are in one row


def findClosestFace(testW, trainW):
     # calculates the distances between testing faces and training faces in the eigenspace
     # and assigns labels to the testing faces based on the closest training face.
     noTest=len(testW)
     noTrain=len(trainW)
     print(noTest,noTrain)
     predFaces=np.zeros(shape=(noTest,1))
     for i in range(noTest):
         min_distance=999999999999999999999999
         min_Train=-1
         for j in range(noTrain-1):
             curr_distance = distance(testW[i], trainW[j])
             if curr_distance < min_distance: # find the closet hydrant
                 min_distance = curr_distance
                 min_Train = j
         predFaces[i]=min_Train
     return predFaces


def distance(p0, p1):
    #""" Calculate the Euclidean distance between two points """
    return math.sqrt((p0[1] - p1[1])**2 + (p0[2] - p1[2])**2)

predFaces=findClosestFace(testW,trainW)
print(predFaces)
index = int(predFaces[0][0])
print(index)


def get_file_path_info(data):
    name_list = []
    a_string='\\'
    name_list += data.split(a_string)
    return name_list

a_string='\\'
print(a_string)

# find closest face name by index
closestFaceImageFile=train_image_names[index]
print(closestFaceImageFile)
closestFaceImage_info=get_file_path_info(closestFaceImageFile)
print(closestFaceImage_info)
closestFaceImage_name = closestFaceImage_info [1].split("_")
closestFaceImage_name_label = closestFaceImage_name [:-1]
print(closestFaceImage_name_label)

# return actual name of test image
print(test_path)
test_info =get_file_path_info(test_path)
print(test_info)
names = test_info[1].split("_")
test_label = names[:-1]
print(test_label)

plt.figure()
plt.gray()

plt.imshow(unknown_face, cmap='gray')
plt.axis('off')
plt.show()

face_image = Image.open(train_image_names[index])
plt.figure()
plt.gray()
plt.imshow(face_image, cmap='gray')
plt.axis('off')
plt.show()

