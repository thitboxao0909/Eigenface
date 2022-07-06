from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
import matplotlib.pyplot as plt
import os
import pca
import math
from PIL import Image
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.neighbors import KNeighborsClassifier


def get_imlist(path):
	# Returns a list of filenames for all jpg images in a directory.
 	return [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg')]

image_names = get_imlist('dataset')

im = np.array(Image.open(image_names[0])) # open one image to get size
height, width = im.shape[0:2] # get the size of the images
imnbr = len(image_names) # get the number of images

# create matrix to store all flattened images
immatrix = np.array([np.array(Image.open(im)).flatten() for im in image_names], 'f')

# perform PCA
eigenvectors, eigenvalues, mean_face = pca.pca(immatrix)

print('heyyyyyyy')
for k in range(eigenvalues.size):
	print(eigenvalues[k])


threhold = 0.85
for k in range(eigenvalues.size):
     R=sum(eigenvalues[:k])/sum(eigenvalues)
     if R > threhold:
        print(k)
        break

threhold = 0.95
for k in range(eigenvalues.size):
     R=sum(eigenvalues[:k])/sum(eigenvalues)
     if R > threhold:
        print(k)
        break


def get_file_path_info(data):
    name_list = []
    a_string='\\'
    name_list += data.split(a_string)
    return name_list

def listToString(s):
    str1=" "     
    return (str1.join(s))

all_image_names =[]

for index in range(len(image_names)):    
    all_file_names =get_file_path_info(image_names[index])
    names = all_file_names[1].split("_")
    test_label = names[:-1]
    final = listToString(test_label)
    all_image_names.append(final)
    
print(all_image_names)

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in split.split(immatrix, all_image_names):
     X_train = immatrix[train_index]
     X_test = immatrix[test_index]
     Y_train = []
     Y_test = []
     for i in range(len(train_index)):
        Y_train.append(all_image_names[train_index[i]])
     for j in range(len(test_index)):
        Y_test.append(all_image_names[test_index[j]])
        
        
print(X_test)
# perform PCA for dimension reduction
eigenvectors, eigenvalues, mean_face = pca.pca(X_train)
k_PCA = 10
U = np.array(eigenvectors[:k_PCA]).transpose()

# normalise face images before projection
normalised_training = X_train - mean_face
normalized_testing = X_test - mean_face

# projection
trainW = np.dot(normalised_training, U)
testW = np.dot(normalized_testing, U)



# normalise features column wise
scaler = StandardScaler()
scaler.fit(trainW)

# apply the transformation to the training and testing data
norm_trainW = scaler.transform(trainW)
norm_testW = scaler.transform(testW)


# build MLP classifier
mlp_learner = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=6000)

print("Fitting the classifier to the training set")
mlp_learner.fit(norm_trainW, Y_train)

print("prediction on the testing set")
Y_pred = mlp_learner.predict(norm_testW)

print("Confusion Matrix")
print(confusion_matrix(Y_test, Y_pred))
print("-" * 40)
print(classification_report(Y_test, Y_pred))
print("Test Accuracy =", round(accuracy_score(Y_test, Y_pred), 4))

# Visualisation
def plot_MLP_results(images, titles, h, w, n_row=3, n_col=3):
     plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
     plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
     for i in range(n_row * n_col):
         plt.subplot(n_row, n_col, i + 1)
         plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
         plt.title(titles[i], size=12)
         plt.xticks(())
         plt.yticks(())
def title_pred_true_labels(pred_labels, true_labels):
     titles = []
     for i in range(len(pred_labels)):
         info = 'predicted: %s\ntrue: %s' % (pred_labels[i], true_labels[i])
         titles.append(info) # remove ID
     return titles
     
results_titles = title_pred_true_labels(Y_pred, Y_test)

plot_MLP_results(X_test, results_titles, height, width)

plt.show()




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

numNearestNeighbors=3
neighbours_id = findKClosestFaces(testW[2], trainW, numNearestNeighbors)


def plot_KNN_results(test_image, test_label, train_images, train_labels, neighbours_id, h, w, n_row=1, n_col=4):
     plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
     plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
     plt.subplot(n_row, n_col, 1)
     plt.imshow(test_image.reshape((h, w)), cmap=plt.cm.gray)
     plt.title('query image true: %s' % (test_label), size=12)
     for i in range(len(neighbours_id)):
         plt.subplot(n_row, n_col, i + 2)
         plt.imshow(train_images[neighbours_id[i]].reshape((h, w)), cmap=plt.cm.gray)
         plt.title(train_labels[neighbours_id[i]], size=12)
         plt.xticks(())
         plt.yticks(())
     
     

plot_KNN_results(X_test[2], Y_test[2], X_train, Y_train, neighbours_id, height, width, n_row=1, n_col=numNearestNeighbors+1)
plt.show()

knn = KNeighborsClassifier(n_neighbors=3)
print(knn)