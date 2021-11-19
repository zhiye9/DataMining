"""
Homework : Evaluating classifiers
Course   : Data Mining (636-0018-00L)

Auxiliary functions.

This file implements the metrics that are invoked from the main program.
"""
# Author: Damian Roqueiro <damian.roqueiro@bsse.ethz.ch>
# October 2015

import numpy as np
import math

"""
Function: confusion_matrix
Returns a 2-by-2 matrix with the counts of TP, FP, TN, FN
The layout is the following:
                      yTrue
               |  y = 1 | y = -1 |
        --------------------------
        y = 1  |   TP   |   FP   |
 yPred  --------------------------
        y = -1 |   FN   |   TN   |
        --------------------------
"""
def confusion_matrix(y_true, y_pred):
    # Create the confusion matrix
    mat = np.zeros((2, 2))
    # Get the unique elements of the array and iterate through them
    vec_elem = np.unique(y_true)
    for elem in vec_elem:
        idx = (y_true == elem)
        # Determine if it's TP or TN
        if elem > 0:
            # TP
            mat[0, 0] = sum(y_pred[idx] == elem)
            # FN
            mat[1, 0] = sum(y_pred[idx] != elem)
        else:
            # TN
            mat[1, 1] = sum(y_pred[idx] == elem)
            # FP
            mat[0, 1] = sum(y_pred[idx] != elem)

    return mat

"""
Function: compute_precision
precision = TP / (TP + FP)
Invoke confusion_matrix() to obtain the counts
"""
def compute_precision(y_true, y_pred):
    mat = confusion_matrix(y_true, y_pred)
    # Divide by the sum of the first row
    return mat[0, 0] / mat.sum(axis=1)[0]

"""
Function: compute_recall
recall = TP / (TP + FN)
Invoke confusion_matrix() to obtain the counts
"""
def compute_recall(y_true, y_pred):
    mat = confusion_matrix(y_true, y_pred)
    # Divide by the sum of the first column
    return mat[0, 0] / mat.sum(axis=0)[0]

"""
Function: compute_accuracy
accuracy = (TP + TN) / (TP + TN + FP + FN)
Invoke the confusion_matrix() to obtain the counts
"""
def compute_accuracy(y_true, y_pred):
    mat = confusion_matrix(y_true, y_pred)
    # Divide by the total sum
    return (mat[0, 0] + mat[1, 1]) / mat.sum()

"""
Function: compute_tp_rate
tp_rate = recall
"""
def compute_tp_rate(y_true, y_pred):
    return compute_recall(y_true, y_pred)

"""
Function: compute_fp_rate
fp_rate = FP / (FP + TN)
Invoke confusion_matrix() to obtain the counts
"""
def compute_fp_rate(y_true, y_pred):
    mat = confusion_matrix(y_true, y_pred)
    # Divide by the sum of the second column
    return mat[0, 1] / mat.sum(axis=0)[1]


