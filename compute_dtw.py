#!/usr/bin/env python2
"""
Homework  : Similarity measures on sets
Course    : Data Mining (636-0018-00L)

Compute all pairwise DTW and euclidean distances of time-series within and between groups.
"""
# Author: Xiao He <xiao.he@bsse.ethz.ch>
# September 2015

import os
import sys
import argparse
import numpy as np
from dynamic_time_warping import DTW
from sklearn.metrics import pairwise_distances as pdist

#------------------------------------------------------------------------------
# Main program
#------------------------------------------------------------------------------

# Set up the parsing of command-line arguments
parser = argparse.ArgumentParser(description="Compute distance functions on time-series")
parser.add_argument("--datadir", required=True,
                    help="Path to input directory containing the EGC200_TRAIN.txt")
parser.add_argument("--outdir", required=True,
                    help="Path to the output directory where timeseries_output.txt will be created")
args = parser.parse_args()

# Set the paths

data_dir = args.datadir
out_dir = args.outdir

# Create the output directory if it does not exist

if not os.path.exists(args.outdir):
    os.makedirs(args.outdir)

# Read the file
data = np.loadtxt("%s/%s" % (args.datadir, 'ECG200_TRAIN.txt'), delimiter=',')

# Create the output file
try:

    file_name = "%s/strings_output.txt" % args.outdir
    f_out = open(file_name, 'w')
except IOError:
    print("Output file %s cannot be created" % file_name)
    sys.exit(1)

cdict = {}
cdict['abnormal'] = -1
cdict['normal'] = 1
lst_group = ['abnormal', 'normal']

# Iterate through all combinations of pairs
for idx_g1 in range(len(lst_group)):
    for idx_g2 in range(idx_g1, len(lst_group)):
        # Get the group data
        group1 = data[data[:, 0] == cdict[lst_group[idx_g1]]]
        group2 = data[data[:, 0] == cdict[lst_group[idx_g2]]]
        # Get average similarity
        count = 0
        D = 0
        E = 0
        vec_sim = np.zeros(2)
        for m in range(0,len(group1)):
            for n in range(0,len(group2)):
                D += DTW(group1[m,range(1,group1.shape[1])], group2[n,range(1,group1.shape[1])])
                E += np.sqrt(sum ((group1[m,range(1,group1.shape[1])]-group2[n,range(1,group1.shape[1])])*(group1[m,range(1,group1.shape[1])]-group2[n,range(1,group1.shape[1])])))
        vec_sim[0] = D/(len(group1)*len(group2))
        vec_sim[1] = E/(len(group1)*len(group2))
        
        '''
        Insert code here!!!
        '''

        # Transform the vector of distances to a string
        str_sim = "\t".join("%.2f" % x for x in vec_sim)

        # Save the output
        f_out.write("%s:%s\t%s\n" % (lst_group[idx_g1], lst_group[idx_g2], str_sim))

f_out.close()
