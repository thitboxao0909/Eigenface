from PIL import Image
from scipy.ndimage import measurements, morphology
import matplotlib.pyplot as plt
import numpy as np
def pca(X):
    #""" Principal Component Analysis
    #input: X, matrix with training data stored as flattened arrays in rows
    #return: projection matrix (with important dimensions first), variance
    #and mean."""
     # get dimensions
     num_data, dim = X.shape
     
     # center data
     mean_X = X.mean(axis=0)
     X = X - mean_X
     
     if dim>num_data:
         # PCA - compact trick used
         M = np.dot(X,X.T) # covariance matrix
         e,EV = np.linalg.eigh(M) # eigenvalues and eigenvectors
         tmp = np.dot(X.T,EV).T # this is the compact trick
         V = tmp[::-1] # reverse since last eigenvectors are the ones we want
         S = np.sqrt(e)[::-1] # reverse since eigenvalues are in increasing order
         for i in range(V.shape[1]):
            V[:,i] /= S
     else:
         # PCA - SVD used
         U, S, V = np.linalg.svd(X)
         V = V[:num_data] # only makes sense to return the first num_data
         
     # return the projection matrix, the variance and the mean
     return V, S, mean_X