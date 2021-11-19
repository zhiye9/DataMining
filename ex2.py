#!/usr/bin/env python2
"""
Solution to Homework 4: Logistic Regression and Decision Trees, 
Part 2: Decision Trees

Date: 13/11/17
Author: Anja Gumpinger, Dean Bodenham
"""

def do_split(data, y, a_idx, theta):
    """
    split the data according to a_idx (feature) and theta (threshold).
    """
    k1 = data[:,a_idx] < theta
    X1 = data[k1]
    k2 = data[:,a_idx] >= theta
    X2 = data[k2]
    
    y1 = y[k1]
    y2 = y[k2]
    
    return(X1,X2,y1,y2)
    

def compute_info_content(y):
    """
    commpute information content info(D).
    """
    
    yl = y.tolist()
    infocontent = 0.0
    
    for i in range(0,3):
        pi = float(yl.count(i))/float(len(y))
        if pi == 0:
            logpi = 0
        else:
            logpi = np.log2(pi)
        infocontent = infocontent + pi*logpi

    infoc = -infocontent
    return(infoc)


def compute_info_a(data, y, a_idx, theta):
    """
    compute conditional information content Info_A(D).
    """

    # Fill in here.
    y1 = do_split(data, y, a_idx, theta)[2]
    y2 = do_split(data, y, a_idx, theta)[3]
    info1 = compute_info_content(y1)
    info2 = compute_info_content(y2)
    infoa = (float(len(y1))/float(len(y)))*info1 + (float(len(y2))/float(len(y)))*info2
    
    return(infoa)


def compute_info_gain(data, y, a_idx, theta):
    """
    compute information gain(A) = Info(D) - Info_A(D)
    """
    
    # Fill in here.
    infoD =  compute_info_content(y)
    infoaD = compute_info_a(data, y, a_idx, theta)
    gain = infoD - infoaD

    return(gain)

import numpy as np
from sklearn.datasets import load_iris


if __name__ == '__main__':

    # Fill in here.
    # to load the data into X and labels into y
    iris = load_iris()
    X = iris.data
    y = iris.target
    # to see feature names and label names:
    featureNames = iris.feature_names
    yNames = iris.target_names
    
    print('\nExercise 2.b\n------------------')
    print('Split ( speal length (cm) < 5.5 ): Information gain = %s' % compute_info_gain(X, y, 0, 5.5))
    print('Split ( speal width  (cm) < 3.0 ): Information gain = %s' % compute_info_gain(X, y, 1, 3.0))
    print('Split ( speal length (cm) < 2.0 ): Information gain = %s' % compute_info_gain(X, y, 2, 2.0))
    print('Split ( speal width  (cm) < 1.0 ): Information gain = %s' % compute_info_gain(X, y, 3, 1.0))
    print('\nExercise 2.c\n------------------')
    print('I would choose speal length and speal width, because they have the largest information gain, and we can observe that it classify the first species successfully. ')