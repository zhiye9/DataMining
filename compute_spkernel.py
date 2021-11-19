#!/usr/bin/env python2
"""
Homework  : Similarity measures on sets
Course    : Data Mining (636-0018-00L)

Compute all pairwise shortest path kernel of graphs within and between groups.
"""
# Author: Xiao He <xiao.he@bsse.ethz.ch>
# September 2015

import os
import sys
import argparse
import numpy as np
import scipy.io
from shortest_path_kernel import floyd_warshall
from shortest_path_kernel import spkernel

#------------------------------------------------------------------------------
# Main program
#------------------------------------------------------------------------------

# Set up the parsing of command-line arguments
parser = argparse.ArgumentParser(description="Compute distance functions on time-series")
parser.add_argument("--datadir", required=True,
                    help="Path to input directory containing the MUTAG.mat")
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
mat = scipy.io.loadmat('../data/MUTAG.mat')
label = np.reshape(mat['lmutag'], (len(mat['lmutag'], )))
adj = np.reshape(mat['MUTAG']['am'], (len(label), ))

# Create the output file
try:
    file_name = "%s/graphs_output.txt" % args.outdir
    f_out = open(file_name, 'w')
except IOError:
    print("Output file %s cannot be created" % file_name)
    sys.exit(1)

sp = np.array([floyd_warshall(A) for A in adj])

cdict = {}
cdict['mutagenic'] = 1
cdict['non-mutagenic'] = -1
lst_group = ['mutagenic', 'non-mutagenic']

# Iterate through all combinations of pairs
for idx_g1 in range(len(lst_group)):
    for idx_g2 in range(idx_g1, len(lst_group)):
        # Get the group data
        group1 = sp[label == cdict[lst_group[idx_g1]]]
        group2 = sp[label == cdict[lst_group[idx_g2]]]

        # Get average similarity
        vec_sim = np.zeros(1)
        
        Sum = 0
        for m in group1:
            for n in group2:
                Sum += spkernel(m[1:], n[1:])
        vec_sim[0] = Sum/(len(group1)*len(group2))
        '''
        Insert code here!!!
        '''

        # Transform the vector of distances to a string
        str_sim = "\t".join("%.2f" % x for x in vec_sim)

        # Save the output
        f_out.write("%s:%s\t%s\n" % (lst_group[idx_g1], lst_group[idx_g2], str_sim))

f_out.close()
