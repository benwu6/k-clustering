import scipy
from scipy import stats
from helper_functions import loaddata, visualize_knn_2D, visualize_knn_images, plotfaces, visualize_knn_boundary
import time
from scipy.io import loadmat
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
# functions that may be helpful
from scipy.stats import mode
import sys
# </GRADED>
%matplotlib notebook
# <GRADED>
# </GRADED>

print('You\'re running python %s' % sys.version.split(' ')[0])

xTr, yTr, xTe, yTe = loaddata("faces.mat")

plt.figure()
plotfaces(xTr[:9, :])


def l2distance(X, Z=None):
    """
    function D=l2distance(X,Z)

    Computes the Euclidean distance matrix.
    Syntax:
    D=l2distance(X,Z)
    Input:
    X: nxd data matrix with n vectors (rows) of dimensionality d
    Z: mxd data matrix with m vectors (rows) of dimensionality d

    Output:
    Matrix D of size nxm
    D(i,j) is the Euclidean distance of X(i,:) and Z(j,:)

    call with only one input:
    l2distance(X)=l2distance(X,X)
    """

    if Z is None:
        Z = X

    n, d1 = X.shape
    m, d2 = Z.shape
    assert (d1 == d2), "Dimensions of input vectors must match!"
    # Your code goes here ..
    S = np.tile(np.atleast_2d(
        np.sum(np.square(X), axis=1)).transpose(), (1, m))
    R = np.tile(np.sum(np.square(Z), axis=1).transpose(), (n, 1))
    G = np.matmul(X, Z.transpose())
    D = S + R - G - G
    D[D < 0] = 0.0
    D = np.sqrt(D)

    return D


def findknn(xTr, xTe, k):
    """
    function [indices,dists]=findknn(xTr,xTe,k);

    Finds the k nearest neighbors of xTe in xTr.

    Input:
    xTr = nxd input matrix with n row-vectors of dimensionality d
    xTe = mxd input matrix with m row-vectors of dimensionality d
    k = number of nearest neighbors to be found

    Output:
    indices = kxm matrix, where indices(i,j) is the i^th nearest neighbor of xTe(j,:)
    dists = Euclidean distances to the respective nearest neighbors
    """
    # Enter your code here
    dists = l2distance(xTr, xTe)
    indices = np.argsort(dists, axis=0)
    indices = indices[: k, :]
    dists = np.sort(dists, axis=0)
    dists = dists[: k, :]
    return indices, dists


def analyze(kind, truth, preds):
    """
    function output=analyze(kind,truth,preds)         
    Analyses the accuracy of a prediction
    Input:
    kind=
        'acc' accuracy, or 
        'abs' absolute loss
    (other values of 'kind' will follow later)
    """

    truth = truth.flatten()
    preds = preds.flatten()

    if kind == 'abs':
        # compute the absolute difference between truth and predictions
        output = np.sum(abs(truth - preds))
        output = output/len(truth)
    elif kind == 'acc':
        output = np.sum(truth != preds)
        output = 1 - output/len(truth)

    return output


def knnclassifier(xTr, yTr, xTe, k):
    """
    function preds=knnclassifier(xTr,yTr,xTe,k);

    k-nn classifier 

    Input:
    xTr = nxd input matrix with n row-vectors of dimensionality d
    xTe = mxd input matrix with m row-vectors of dimensionality d
    k = number of nearest neighbors to be found

    Output:

    preds = predicted labels, ie preds(i) is the predicted label of xTe(i,:)
    """
    # fix array shapes
    yTr = yTr.flatten()
    # Your code goes here
    indices, dist = findknn(xTr, xTe, k)
    labels = yTr[indices]
    preds = stats.mode(labels).mode
    preds = preds.flatten()
    return preds


print("Face Recognition: (1-nn)")
xTr, yTr, xTe, yTe = loaddata("faces.mat")  # load the data
t0 = time.time()
preds = knnclassifier(xTr, yTr, xTe, 1)
result = analyze("acc", yTe, preds)
t1 = time.time()
print("You obtained %.2f%% classification acccuracy in %.4f seconds\n" %
      (result*100.0, t1-t0))

matplotlib notebook
visualize_knn_boundary(knnclassifier)
