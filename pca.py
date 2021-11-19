#!/usr/bin/python

"""
Homework: Principal Component Analysis
Course  : Data Mining II (636-0019-00L)
"""
import scipy as sp
import scipy.linalg as linalg
import pylab as pl
import numpy as np

from utils import plot_color

'''############################'''
'''Principle Component Analyses'''
'''############################'''

'''
Compute Covariance Matrix
Input: Matrix of size #samples x #features
Output: Covariance Matrix of size #features x #features
Note: Do not use scipy or numpy cov. Implement the function yourself.
      You can of course add an assert to check your covariance function
      with those implemented in scipy/numpy.
'''

def computeCov(X=None):
	# Please fill this function
    mean = np.zeros(shape=(len(X[0]),1))
    Y = X
    Z = np.zeros(shape=(len(X[0]),len(X[0])))
    for j in range(len(X[0])):
        mean[j] = sum(X[:,j])/len(X)
        for i in range(len(X)):
            Y[i,j] = X[i,j]-mean[j]
    
    for i in range(len(Z)):
        for j in range(len(Z[0])):
            Z[i,j] = sum(Y[:,i]*Y[:,j])/(len(X)-1)
            
    return Z,Y           


'''
Compute PCA
Input: Covariance Matrix
Output: [eigen_values,eigen_vectors] sorted in such a why that eigen_vectors[:,0] is the first principle component
        eigen_vectors[:,1] the second principle component etc...
Note: Do not use an already implemented PCA algorithm. However, you are allowed to use an implemented solver 
      to solve the eigenvalue problem!
'''



def computePCA(matrix=None):
	# Please fill this function
    eva,eve = np.linalg.eig(matrix) 
    evas = eva[np.argsort(-eva)]
    eves = eve.T[np.argsort(-eva)].T
    
    return evas, eves
    
'''
Transform Input Data Onto New Subspace
Input: pcs: matrix containing the first x principle components
       data: input data which should be transformed onto the new subspace
Output: transformed input data. Should now have the dimensions #samples x #components_used_for_transformation
'''

def transformData(pcs=None,data=None):
    # Please fill this function
    newdata = np.dot(data,pcs)
    
    return newdata

'''
Compute Variance Explaiend
Input: eigen_values
Output: return vector with varianced explained values. Hint: values should be between 0 and 1 and should sum up to 1.
'''


def computeVarianceExplained(evals=None):
    # Please fill this function
    total = sum(evals)
    valex = np.zeros((len(evals),1))
    for i in range(len(evals)):
        valex[i] = sorted(evals, reverse=True)[i] / total
    
    return valex
    

'''############################'''
'''Different Plotting Functions'''
'''############################'''

'''
Plot Cumulative Explained Variance
Input: var: variance explained vector
       filename: filename to store the file
''' 


def plotCumSumVariance(var=None,filename="cumsum.pdf"):
    #PLOT FIGURE
    #You can use plot_color[] to obtain different colors for your plots
    #Save file
    cumvar = var.cumsum()

    pl.figure()
    pl.bar(np.arange(len(var)), cumvar)
    pl.axhline(y=0.9, color='r', linestyle='-')
    pl.savefig(filename)
'''
Plot Transformed Data
Input: transformed: data matrix (#sampels x 2)
       labels: target labels, class labels for the samples in data matrix
       filename: filename to store the plot
'''


    
def plotTransformedData(transformed=None,labels=None,filename="exercise1.pdf"):
    #PLOT FIGURE
    #You can use plot_color[] to obtain different colors for your plots
    #Save File

    pl.figure()
    unilab = np.unique(labels)
    for i in range(len(unilab)):
        typex = transformed[labels==unilab[i],:][:,0]
        typey = transformed[labels==unilab[i],:][:,1]
        pl.scatter(typex, typey, label = unilab[i],c = plot_color[i])
        pl.legend()
    
    pl.savefig(filename)
'''############################'''
'''Data Preprocessing Functions'''
'''############################'''

'''
Exercise 2
Data Normalisation (Zero Mean, Unit Variance)
'''


def dataNormalisation(X=None):
    # Please fill this function
    mean = np.zeros(shape=(len(X[0]),1))
    A = np.zeros(shape=(len(X),len(X[0])))
    for j in range(len(X[0])):
        mean[j] = sum(X[:,j])/len(X)
        for i in range(len(X)):
            A[i,j] = X[i,j]-mean[j]
    
    std = np.std(A, axis=0)
    B = np.zeros(shape=(len(X),len(X[0])))
    for m in range(len(X[0])):
        for n in range(len(X)):
            B[n,m] = A[n,m]/std[m]
   
    
    return B
