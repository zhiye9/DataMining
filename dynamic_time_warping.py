#!/usr/bin/env python2

import numpy as np

def DTW(t1, t2):
    W = np.zeros((len(t2),len(t1)))
    for i in range(len(t2)):
        for j in range(len(t1)):
            W[i,j] = abs(t2[i]-t1[j])
            
    C = np.zeros((len(t2)+1,len(t1)+1))
    C[0,:] = C[:,0] = float("inf")
    C[0,0] = 0.0
    for i in range(1,len(t2)+1):
        for j in range(1,len(t1)+1):
            C[i,j] = min(W[i-1,j-1]+C[i-1,j],W[i-1,j-1]+C[i-1,j-1],W[i-1,j-1]+C[i,j-1])
    
    return C[len(t2),len(t1)]
