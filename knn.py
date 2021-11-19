"""
Homework : k-Nearest Neighbor and Naive Bayes
Course   : Data Mining (636-0018-00L)

Main program for k-NN.
Predicts the labels of the test data using the training data.
The k-NN algorithm is executed for different values of k (user-entered parameter)

"""
# Author: Damian Roqueiro <damian.roqueiro@bsse.ethz.ch>
# October 2015

import argparse
import os
import sys
import numpy as np
# Import the file with the peformance metrics created for a previous homework
import classif_eval

# Class imports
from classes.knn_classifier import KNNClassifier

# Constants
# 1. Files with the datapoints and class labels
DATA_FILE  = "matrix_mirna_input.txt"
PHENO_FILE = "phenotype.txt"

# 2. Classification performance metrics to compute
PERF_METRICS = ["accuracy", "precision", "recall"]

"""
Function load_data
Receives the path to a directory that will contain the DATA_FILE and PHENO_FILE.
Loads both files into memory as numpy arrays. Matches the patientId to make
sure the class labels are correctly assigned.

Returns
 X : a matrix with the data points
 y : a vector with the class labels
"""
def load_data(dir_path):
    # Read PHENO_FILE and create a dictionary with the patientId as key
    try:
        file_name = "%s/%s" % (dir_path, PHENO_FILE)
        f_in = open(file_name, 'r')
    except IOError:
        print(("Input file %s cannot be opened" % file_name))
        sys.exit(1)
    pheno_dict = {}
    # Read the file, line by line. Ignore the header
    f_in.readline()
    for line in f_in:
        # Separate the fields
        parts = line.rstrip().split("\t")
        # Set the values in the dictionary (key=patientId)
        # '+' is positive class
        # '-' is negative class
        pheno_dict[parts[0]] = 1 if parts[1] == "+" else -1

    # Read DATA_FILE
    try:
        file_name = "%s/%s" % (dir_path, DATA_FILE)
        f_in = open(file_name, 'r')
    except IOError:
        print(("Input file %s cannot be opened" % file_name))
        sys.exit(1)

    X_tmp = []
    y_tmp = []
    # Read the file, line by line. Ignore the header
    f_in.readline()
    for line in f_in:
        # Separate the fields
        parts = line.rstrip().split("\t")

        # Append the data point to the temporary list
        X_tmp.append(parts[1:len(parts)])
        # Get the phenotype for the current patient
        y_tmp.append(pheno_dict[parts[0]])

    # Convert to numpy arrays and return
    X = np.array(X_tmp, dtype=np.float)
    y = np.array(y_tmp)

    return X, y

"""
Function obtain_performance_metrics
Receives two numpy arrays with the true and predicted labels.
Computes all classification performance metrics

Returns a vector with one value per metric. The positions in the
vector match the metric names in PERF_METRICS.
"""
def obtain_performance_metrics(y_true, y_pred):
    # Initialize the vector that will contain a value per metric
    vec = np.zeros(len(PERF_METRICS))

    # Iterate through all metrics and compute them
    for i, metric in enumerate(PERF_METRICS):
        # Dynamically get the function to call
        performance_fn = getattr(classif_eval, "compute_%s" % metric)

        # Compute the metric
        vec[i] = performance_fn(y_true, y_pred)

    return vec


#------------------------------------------------------------------------------
# Main program
#------------------------------------------------------------------------------

# Set up the parsing of command-line arguments
parser = argparse.ArgumentParser(description="Compute distance functions on vectors")
parser.add_argument("--traindir", required=True,
                    help="Path to the location of the training data. It will contain the files: \
                          matrix_mirna_input.txt and phenotype.txt.")
parser.add_argument("--testdir", required=True,
                    help="Path to the location of the test data. It will contain the files: \
                          matrix_mirna_input.txt and phenotype.txt.")
parser.add_argument("--outdir", required=True,
                    help="Path to the output directory, where the output file will be created")
parser.add_argument("--mink", required=True,
                    help="Minimum value of k.")
parser.add_argument("--maxk", required=True,
                    help="Maximum value of k. In conjunction with --mink, it provides a range of values to run k-NN.")
args = parser.parse_args()

# If the output directory does not exist, then create it
if not os.path.exists(args.outdir):
    os.makedirs(args.outdir)

# Create the output file
try:
    file_name = "%s/output_knn.txt" % args.outdir
    f_out = open(file_name, 'w')
except IOError:
    print(("Output file %s cannot be created" % file_name))
    sys.exit(1)

# Read the training and test data. For each dataset, get also the true labels.
# Important: Match the patientId between data points and class labels
# Training
X_train, y_train = load_data(args.traindir)
# Test
X_test, y_test = load_data(args.testdir)

# Create the k-NN object
knn_obj = KNNClassifier(X_train, y_train, metric='euclidean')
# Set the verbose level (for debugging purposes)
knn_obj.debug('off')

# Save the header to the output file
f_out.write("Value of k\tAccuracy\tPrecision\tRecall\n")

# Iterate through all possible values of k
for k in range(int(args.mink), int(args.maxk) + 1):
    # Set the number of neighbors for the object
    knn_obj.set_k(k)
    vec_perf = [] ##Remove this when done

    # For each value of k, iterate through all data points in the test data
    # Create an empty vector where the predictions will be stored. It will have
    # the same length as Y_train
    # Compute the performance metrics
    '''
    Insert code here!!!
    '''

    # Transform the vector of measures to a string
    str_perf = "\t".join("%.2f" % x for x in vec_perf)

    # Save the output
    f_out.write("%d\t%s\n" % (k, str_perf))

f_out.close()

