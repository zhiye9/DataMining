"""
Homework: k-NN and Naive Bayes
Course  : Data Mining (636-0018-00L)

Partition a dataset into training and test.
If more than one file is in the input directory, the program assumes the
rows in the files are paired and partitions them maintaining row integrity.

The percentages for each subset will be user-entered parameters.
They do not need to add up to 1.0
"""
# Author: Damian Roqueiro <damian.roqueiro@bsse.ethz.ch>
# October 2015

import argparse
import sys
import os
import glob
import subprocess
from sklearn import cross_validation

#------------------------------------------------------------------------------
# Main program
#------------------------------------------------------------------------------

# Set up the parsing of command-line arguments
parser = argparse.ArgumentParser(description="Partition the contents of a file (lines) into training and test sets")
parser.add_argument("--seed", required=True,
                    help="Random seed to reproduce results")
parser.add_argument("--datadir", required=True,
                    help="Path to the data directory. All files in this directory will be partitioned")
parser.add_argument("--header", required=True,
                    help="Is there a header line in the file(s) [yes/no]")
parser.add_argument("--ptrain", required=True,
                    help="Percentage of records used for training (e.g. 0.7)")
parser.add_argument("--ptest", required=True,
                    help="Percentage of records used for testing (e.g. 0.3). This is used in case ptrain + ptest < 1.0")
parser.add_argument("--outdir", required=True,
                    help="Path to the output directory. Two subdirectories will be created")
args = parser.parse_args()

# Set the parameters
seed    = int(args.seed)
data_dir= args.datadir
header  = True if args.header == "yes" else False
ptrain  = float(args.ptrain)
ptest   = float(args.ptest)
out_dir = args.outdir

# Check parameter consistency
if ptrain + ptest > 1.0:
    print("Error. Percentages of train and test sets cannot add up to more than 1.0")
    sys.exit(1)

# Create the output directory and the subdirectories if they do not exist
sub_dir = "%s/train" % out_dir
if not os.path.exists(sub_dir):
    os.makedirs(sub_dir)
sub_dir = "%s/test" % out_dir
if not os.path.exists(sub_dir):
    os.makedirs(sub_dir)

# Get the list of files to process
lst_files = glob.glob("%s/*" % data_dir)
# Get the number of lines in one of the input files using an external process
# Note: The assumption is that if many files are to be partitioned, then all of them
#       have the same number of lines and are matched line by line
proc = subprocess.Popen("wc -l %s" % lst_files[0], shell=True, stdout=subprocess.PIPE)
n_lines = int(proc.communicate()[0].split()[0])
if header:
    n_lines = n_lines - 1

# Create the indices of records (lines) that will belong to training or test
rs = cross_validation.ShuffleSplit(n_lines, n_iter=1, train_size=ptrain, test_size=ptest, random_state=seed)

# Read each line in the file and determine to what set it belongs
for train_index, test_index in rs:
    # Iterate through all files in data_dir
    for in_file in lst_files:
        header  = True if args.header == "yes" else False
        # Open the input file
        try:
            f_in = open(in_file, 'r')
        except IOError:
            print(("Cannot read file %s" % in_file))
            sys.exit(1)

        # Create the output files for train and test (use the same name as input)
        base_name = os.path.basename(in_file)
        # Train
        try:
            out_train = "%s/train/%s" % (out_dir, base_name)
            f_train   = open(out_train, 'w')
        except IOError:
            print(("Cannot create output file" % out_train))
            sys.exit(1)
        # Test
        try:
            out_test = "%s/test/%s" % (out_dir, base_name)
            f_test   = open(out_test, 'w')
        except IOError:
            print(("Cannot create output file" % out_test))
            sys.exit(1)

    	# Iterate through the input file, line by line
        k = 0
        for line in f_in:
            if header:
                # There is a header line. Save it to both files
                f_train.write(line)
                f_test.write(line)
                header = False
                continue

            # Determine if this line should go to the train or test subset
            if k in train_index:
                f_train.write(line)
            if k in test_index:
                f_test.write(line)

            # Next k
            k = k + 1

    # Close the files
    f_in.close()
    f_train.close()
    f_test.close()

