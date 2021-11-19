#!/usr/bin/env python2
"""
Solution to Homework 4: Logistic Regression and Decision Trees, 
Part 1: Logistic Regression

Date: 13/11/17
Author: Anja Gumpinger, Dean Bodenham
"""

def computeAccuracy(Y, Yhat):
    """
    compute the accuracy of a prediction Yhat wrt. the true class labels Y
    
    :param Y: true class labels
    :type Y: list
    
    :param Yhat: predicted class labels
    :type Yhat: list
    
    :return: accuracy value
    :rtype: float
    """
    L = len(Y)

    # true/false pos/neg.
    tp_count = 0
    fp_count = 0
    tn_count = 0
    fn_count = 0

    # define positive and negative classes.
    pos = 1
    neg = 0
    for i in range(0, L):
        if Y[i] == pos:
            # positive class.
            if Yhat[i] == pos:
                tp_count += 1
            else:
                fn_count += 1
        else:
            # negative class.
            if Yhat[i] == neg:
                tn_count += 1
            else:
                fp_count += 1

    # compute the accurary.
    accuracy = (tp_count + tn_count) / float(
        tp_count + fp_count + tn_count + fn_count)

    # output to screen.
    print('TP: {0:d}'.format(tp_count))
    print('FP: {0:d}'.format(fp_count))
    print('TN: {0:d}'.format(tn_count))
    print('FN: {0:d}'.format(fn_count))
    print('accuracy: {0:.3f}'.format(accuracy))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


if __name__ in "__main__":

    # Fill in here.
    fileTrain = "diabetesTrain.csv"
    #read data from file using pandas
    df = pd.read_csv(fileTrain)
    # extract first 7 columns to data matrix X (actually, a numpy ndarray)
    X = df.ix[:, 0:7].as_matrix()
    # extract 8th column (labels) to numpy array
    Y = df.ix[:, 7].as_matrix()

    fileTest = "diabetesTest.csv"
    #read data from file using pandas
    df1 = pd.read_csv(fileTest)
    # extract first 7 columns to data matrix X (actually, a numpy ndarray)
    X1 = df1.ix[:, 0:7].as_matrix()
    # extract 8th column (labels) to numpy array
    Y1 = df1.ix[:, 7].as_matrix()

    logistic_model = LogisticRegression()
    logistic_model.fit(X, Y)
    Yhat = logistic_model.predict(X1)
    
    print('\nExercise 1.b\n------------------')
    computeAccuracy(Y1, Yhat)

    print('\nExercise 1.c\n------------------')
    print('For the diabetes dataset I would choose LDA, because it has a higher accuracy, and the total amount of patients is not very large, so the accuracy is reliable')

    print('\nExercise 1.d\n------------------')
    print('For another dataset, it depends on the feature of datasets. For example, if variables are normally distributed, I would choose LDA. And when the classifying groups of samples are relatively few, I would also choose LDA. But when the assumption of LDA failed, I would choose Logistic Regression because it is based on maximum likelihood method. Meanwhile, Logistic Regression performance better in binary classification while LDA performance better in multi-classification.')
