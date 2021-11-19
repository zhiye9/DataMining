"""
Course  : Data Mining II (636-0019-00L)
"""
import scipy as sp
import scipy.linalg as linalg
import pylab as pl
from utils import *
from utils import plot_color, plot_markers

'''
Compute Distance Matrix using Euclideans Distance
Input: matrix of size n x m, where n is the number of samples and m the number of features
Output: distance matrix of size n x n 
'''
def computeEuclideanDistanceMatrix(matrix=None):
    E = sp.spatial.distance.pdist(matrix,'euclidean')
    M = sp.spatial.distance.squareform(E)
    return M

'''
Classical Metric Multidimensional Scaling
Input: matrix: distance matrix of size n x n
       n_components: number of dimensions/components to return
Output: transformed_data of size n x n_components
'''

def classicalMDS(matrix=None,n_components=2):
    A = -0.5*(matrix**2)
    I = np.identity(A.shape[0])
    H = I - np.dot(np.ones([A.shape[0],A.shape[0]]),np.ones([A.shape[0],A.shape[0]]).T)/A.shape[0]
    B = np.dot(np.dot(H,A),H)
    [eigen_values,eigen_vectors] = linalg.eig(B)
    indices = sp.argsort(-eigen_values)
    seigen_values = eigen_values[indices]
    seigen_vectors = eigen_vectors[:,indices]
    sseigen_values = seigen_values[:n_components]
    sseigen_vectors = seigen_vectors[:,:n_components]
    sslamda = np.diag(sseigen_values)
    V = sseigen_vectors
    transformed_data = np.dot(V,sp.real(np.sqrt(sslamda)))
    return transformed_data

'''
Plot cities as dots and add name tag to city
Input: transformed_data: transformed data of size n x 2
       names: array of city_names
       filename: filename to save image
'''

def plotCities(transformed_data=None,names=None,filename="cities.pdf"):
    pl.figure()
    pl.scatter(transformed_data[:,0],transformed_data[:,1],color=plot_color)
    for i, names in enumerate(names):
        pl.annotate(names, (transformed_data[:,0][i],transformed_data[:,1][i]))
    pl.savefig(filename)

        
 

