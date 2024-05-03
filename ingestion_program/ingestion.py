#!/usr/bin/env python
# Copied description and defaults from codabench/iris/bundle/ingestion_program/ingestion.py

# Usage: python ingestion.py input_dir output_dir ingestion_program_dir submission_program_dir

# AS A PARTICIPANT, DO NOT MODIFY THIS CODE.
#
# This is the "ingestion program" written by the organizers.
# This program also runs on the challenge platform to test your code.
#
# The input directory input_dir (e.g. sample_data/) contains the dataset(s), including:
#   dataname/dataname_feat.name          -- the feature names (column headers of data matrix)
# 	dataname/dataname_feat.type          -- the feature type "Numerical", "Binary", or "Categorical" (Note: if this file is abscent, get the feature type from the dataname.info file)
#   dataname/dataname_label.name         -- the label names (column headers of the solution matrix)
# 	dataname/dataname_public.info        -- public information on the dataset
# 	dataname/dataname_test.data          -- training, validation and test data (solutions/target values are given for training data only)
# 	dataname/dataname_train.data
# 	dataname/dataname_train.solution
# 	dataname/dataname_valid.data
#
# The output directory output_dir (e.g. sample_result_submission/)
# will receive the predicted values (no subdirectories):
# 	dataname_test.predict
# 	dataname_valid.predict
#
# The code directory submission_program_dir (e.g. sample_code_submission/) should contain your
# code submission model.py (an possibly other functions it depends upon).
#
# ALL INFORMATION, SOFTWARE, DOCUMENTATION, AND DATA ARE PROVIDED "AS-IS".
# ISABELLE GUYON, CHALEARN, AND/OR OTHER ORGANIZERS OR CODE AUTHORS DISCLAIM
# ANY EXPRESSED OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR ANY PARTICULAR PURPOSE, AND THE
# WARRANTY OF NON-INFRIGEMENT OF ANY THIRD PARTY'S INTELLECTUAL PROPERTY RIGHTS.
# IN NO EVENT SHALL ISABELLE GUYON AND/OR OTHER ORGANIZERS BE LIABLE FOR ANY SPECIAL,
# INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER ARISING OUT OF OR IN
# CONNECTION WITH THE USE OR PERFORMANCE OF SOFTWARE, DOCUMENTS, MATERIALS,
# PUBLICATIONS, OR INFORMATION MADE AVAILABLE FOR THE CHALLENGE.
#
# Main contributors: Isabelle Guyon and Arthur Pesah, March-October 2014
# Lukasz Romaszko April 2015
# Originally inspired by code code: Ben Hamner, Kaggle, March 2013
# Modified by Ivan Judson and Christophe Poulain, Microsoft, December 2013
# Last modifications Isabelle Guyon, October 2017

# =========================== BEGIN OPTIONS ==============================
# Verbose mode:
##############
# Recommended to keep verbose = True: shows various progression messages
verbose = True # outputs messages to stdout and stderr for debug purposes

# Debug level:
##############
# 0: run the code normally, using the time budget of the tasks
# 1: run the code normally, but limits the time to max_time
# 2: run everything, but do not train, generate random outputs in max_time
# 3: stop before the loop on datasets
# 4: just list the directories and program version
debug_mode = 0

# Time budget
#############
# Maximum time of training in seconds PER DATASET (there may be several datasets).
# The code should keep track of time spent and NOT exceed the time limit
# in the dataset "info" file, stored in D.info['time_budget'], see code below.
# If debug >=1, you can decrease the maximum time (in sec) with this variable:
max_time = 500

# Maximum number of cycles, number of samples, and estimators
#############################################################
# Your training algorithm may be fast, so you may want to limit anyways the
# number of points on your learning curve (this is on a log scale, so each
# point uses twice as many time than the previous one.)
# The original code was modified to do only a small "time probing" followed
# by one single cycle. We can now also give a maximum number of estimators
# (base learners).
max_cycle = 1
max_estimators = float('Inf')
max_samples = 50000

# I/O defaults
##############
# If true, the previous output directory is not overwritten, it changes name
save_previous_results = False
# Use default location for the input and output data:
# If no arguments to run.py are provided, this is where the data will be found
# and the results written to. Change the root_dir to your local directory.
root_dir = "../"
default_input_dir = root_dir + "input_data"
default_output_dir = root_dir + "sample_result_submission"
default_program_dir = root_dir + "ingestion_program"
default_submission_dir = root_dir + "baselines/DINO_SGD_code_submission"

# =============================================================================
# =========================== END USER OPTIONS ================================
# =============================================================================

import os
from sys import argv, path

def write_results(outfile, scores):
    """Should write the score to our output directory
        TODO:
    """
    pass

if __name__ == "__main__":
    #### INPUT/OUTPUT: Get input and output directory names
    if len(argv)==1: # Use the default input and output directories if no arguments are provided
        input_dir = default_input_dir
        output_dir = default_output_dir
        program_dir= default_program_dir
        submission_dir= default_submission_dir
    else:
        input_dir = os.path.abspath(argv[1])
        output_dir = os.path.abspath(argv[2])
        program_dir = os.path.abspath(argv[3])
        submission_dir = os.path.abspath(argv[4])
    if verbose:
        print("Using input_dir: " + input_dir)
        print("Using output_dir: " + output_dir)
        print("Using program_dir: " + program_dir)
        print("Using submission_dir: " + submission_dir)
        
    path.append(program_dir) # In order to access libraries from our own code
    path.append(submission_dir) # In order to access libraries of the user
    
    from model import Model
    
    path_to_csv_for_training = None # TODO
    path_to_csv_for_testing = None # TODO
    M = Model()
    M.fit(path_to_csv_for_training)
    scores = M.predict(path_to_csv_for_testing)
    
    write_results(os.path.join(output_dir, "scores.txt"), scores)
