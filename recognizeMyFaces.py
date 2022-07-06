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
 	return [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg')]

train_image_names = get_imlist('Part2_training_images')

im = np.array(Image.open(train_image_names[0])) # open one image to get size
h, w = im.shape[0:2] # get the size of the images
imnbr = len(train_image_names) # get the number of images

# create matrix to store all flattened images
immatrix = np.array([np.array(Image.open(im)).flatten() for im in train_image_names], 'f')

# perform PCA
eigenvectors, eigenvalues, mean_face = pca.pca(immatrix)

def get_file_name(path):
    name_list = []
    dash='\\'
    name_list += path.split(dash)
    return name_list

def listToString(s):
    str=" "     
    return (str.join(s))

all_image_names = [] # Get labels

for index in range(len(train_image_names)):    
    all_file_names =get_file_name(train_image_names[index])
    names = all_file_names[1].split("_")
    test_label = names[:-1]
    final = listToString(test_label)
    all_image_names.append(final)

# show some images 
plt.figure()
plt.gray()
plt.subplot(2,6,1)
plt.title("mean")
plt.imshow(mean_face.reshape(h,w))
plt.axis('off')

# eigenface_titles = ["eigenface %d" % i for i in range(7)]
for i in range(10):
    plt.subplot(2,6,i+2)
    plt.imshow(eigenvectors[i].reshape(h,w))
    plt.title("# %d"%(i+1))
    plt.axis('off')
     
#plt.show()

for k in range(eigenvalues.size):
     R=sum(eigenvalues[:k])/sum(eigenvalues)
     print(eigenvalues[k])
     print('R: ', R)

threshold = 0.85
for k in range(eigenvalues.size):
     R=sum(eigenvalues[:k])/sum(eigenvalues)
     if R > threshold:
        print(k)
        break

threshold = 0.95
for k in range(eigenvalues.size):
     R=sum(eigenvalues[:k])/sum(eigenvalues)
     if R > threshold:
        print(k)
        break

normalised_training = np.ndarray(shape=(len(train_image_names), h*w))

for i in range(len(train_image_names)):
    normalised_training[i] = np.subtract(immatrix[i],mean_face)

U = np.array(eigenvectors[:k]).transpose()
trainW = np.dot(normalised_training, U)

test_path1= 'Part2_testing_images/hoang_wink.jpg'
unknown_face1 = Image.open(test_path1)
test_image1 = np.array(unknown_face1, dtype='float64').flatten()
normalised_test1 = np.subtract(test_image1, mean_face)
testW1 = np.dot(normalised_test1, U)
testW1=testW1.reshape(1, testW1.shape[0])

test_path2= 'Part2_testing_images/hoang_surprise.jpg'
unknown_face2 = Image.open(test_path2)
test_image2 = np.array(unknown_face2, dtype='float64').flatten()
normalised_test2 = np.subtract(test_image2, mean_face)
testW2 = np.dot(normalised_test2, U)
testW2=testW2.reshape(1, testW2.shape[0])


def findKClosestFaces(a_test, trains, K):
     # calculates the distances between a testing face and all the training faces in the eigenspace
     # arrenge the calculated distances in ascending order
     # find K closest training faces, and return the id.
     noTrain = len(trainW)
     distances = list()
     for i in range(noTrain):
         dist = distance(a_test, trains[i])
         distances.append((dist, i))
     distances.sort(key=lambda x: x[0])
     
     neighbors = list()
     for k in range(K):
        neighbors.append(distances[k][1])
     return neighbors

def distance(pointA, pointB):
     dist = np.linalg.norm(pointA - pointB)
     return dist

numNearestNeighbors=5
neighbours_id1 = findKClosestFaces(testW1, trainW, numNearestNeighbors)
neighbours_id2 = findKClosestFaces(testW2, trainW, numNearestNeighbors)


def plot_KNN_results(test_image, test_label, train_images, train_labels, neighbours_id, h, w, n_row=1, n_col=4):
     plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
     plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
     plt.subplot(n_row, n_col, 1)
     plt.imshow(test_image.reshape((h, w)), cmap=plt.cm.gray)
     plt.title('test image: %s' % (test_label), size=12)
     for i in range(len(neighbours_id)):
        plt.subplot(n_row, n_col, i + 2)
        plt.imshow(train_images[neighbours_id[i]].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(train_labels[neighbours_id[i]], size=12)
        plt.xticks(())
        plt.yticks(())

plot_KNN_results(test_image1, "hoang", immatrix, all_image_names, neighbours_id1, h, w, n_row=1, n_col=numNearestNeighbors+1)
plot_KNN_results(test_image2, "hoang", immatrix, all_image_names, neighbours_id2, h, w, n_row=1, n_col=numNearestNeighbors+1)
plt.show()
