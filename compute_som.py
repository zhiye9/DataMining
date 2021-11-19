# -*- coding: utf-8 -*-
"""
Homework: Self-organizing maps
Course  : Data Mining II (636-0019-00L)

Auxiliary functions to help in the implementation of an online version
of the self-organizing map (SOM) algorithm.
"""
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import pandas as pd
import sys
import argparse
import os
from somutils import *



parser = argparse.ArgumentParser()
parser.add_argument("--exercise", required=True)
parser.add_argument("--outdir", required=True)
parser.add_argument("--p", required=True)
parser.add_argument("--q", required=True)
parser.add_argument("--N", required=True)
parser.add_argument("--alpha_max", required=True)
parser.add_argument("--epsilon_max", required=True)
parser.add_argument("--file", required=False)
args = parser.parse_args()

    
if args.exercise == "1":
    #Exercise1.a
    X = makeSCurve()
    buttons, grid, error = SOM(X, int(args.p), int(args.q), int(args.N), int(args.alpha_max), int(args.epsilon_max), compute_error=True)
    #Exercise1.b
    plotDataAndSOM(X, buttons, fileName = "exercise 1b.pdf")
    #Exercise1.c
    plt.figure()
    plt.plot(range(100),error, c='red')
    plt.xlabel('Iteration')
    plt.ylabel('Reconstruction error')
    plt.savefig('exercise 1c.pdf')
   
if args.exercise == "2":
    #Exercise2.a
    crab = pd.read_csv("%s" % args.file, sep='\t')
    crab1 = crab.drop(['FL','RW','CL','CW','BD'],axis=1)  
    crabdata = pd.read_csv('crabs.txt', sep='\t').iloc[0:200,[3,4,5,6,7]].values
    Cbuttons, Cgrid, Cerror = SOM(crabdata, int(args.p), int(args.q), int(args.N), int(args.alpha_max), int(args.epsilon_max), compute_error=False)

    buttons = initButtons(crabdata,  int(args.p)*int(args.q))
    index = []
    for i in range(crabdata.shape[0]):
        index = np.append(index, findNearestButtonIndex(crabdata[i], buttons))
    crab1['label'] = index  
    crab1.to_csv("%s/output som crabs.txt" % args.outdir, sep = ' ',header=True, index=False)    
    #Exercise2.b 
    idInfo = crab1  
    plotSOMCrabs(crabdata, idInfo, Cgrid, Cbuttons, fileName = 'exercise 2b.pdf')


