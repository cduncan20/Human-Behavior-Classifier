import numpy as np
import pandas as pd
import sys
import os

# Import third party libraries
sys.path.append(os.path.abspath('libsvm-3.24/python'))
from svmutil import *
sys.path.append(os.path.abspath('libsvm-3.24/tools'))
from grid import *


def grid_search(reps_saved_folder, rep_file_name):
    # Specify path to perform grid search
    rep_path = 'project3/output_data/'+ reps_saved_folder + '/' + rep_file_name

    # Perform grid search on data
    rate, param = find_parameters(rep_path, '-png ' + rep_path + '.png -out ' + rep_path + '.out')
    print("")


def train_test_model(d2_path, d2t_path, c, gamma):
	# Define training hyperparameters & Kernel
	param = svm_parameter("-q -c {0} -g {1}".format(c, gamma))  # Hyperparameters
	param.kernel_type = RBF  # Kernel

    # Read in training data and train model
	train_labels, train_input = svm_read_problem(d2_path)  # Read in training data
	prob = svm_problem(train_labels, train_input)  # Generate problem
	model = svm_train(prob, param)  # Train model

	# Read in test data and test model
	test_labels, test_input = svm_read_problem(d2t_path)  # Read in testing data
	pred_labels, pred_acc, pred_vals = svm_predict(test_labels, test_input, model)  # Test model

	# Generates a confusion matrix for reference
	conf_mat = confusion_matrix(pred_labels, test_labels)
	print(conf_mat)
	print("")


def confusion_matrix(pred_labels, true_labels):
	''' Generates the confusion matrix for a list of predicted labels vs ground truth labels 

	Args:
		pred_labels (np.ndarray): a list of predicted labels for a set of data
		true_labels (np.ndarray): a list of ground truth labels for a set of data that corresponds to pred_labels

	Returns:
		confusion_matrix (pd.DataFrame): a labeled DataFrame that encodes the confusion matrix
	'''
    # Create list of actions from predicted labels
	pred_actions = []
	for action in pred_labels:
		if action not in pred_actions:
			pred_actions.append(action)
	pred_actions.sort()

	# Create list of actions from true labels
	true_actions = []
	for action in true_labels:
		if action not in true_actions:
			true_actions.append(action)	
	true_actions.sort()

    # Create confusion matrix
	confusion_matrix = np.zeros((len(true_actions),len(pred_actions)))  # Initialize matrix
	true_vals_counter = 0
	for pred_val in pred_labels:
        # Index of predicted value within list of possible actions
		pred_action_index = pred_actions.index(pred_val)

        # Retrieve the true_val within true_labels at same index of pred_val in pred_labels
		true_val = true_labels[true_vals_counter]

        # Index of true value within list of possible actions
		true_action_index = true_actions.index(true_val)

        # Add value of 1 to confusion matrix at element (true_action_index, pred_action_index)
		confusion_matrix[true_action_index, pred_action_index] = confusion_matrix[true_action_index, pred_action_index] + 1
		true_vals_counter += 1

    # Convert confusion_matrix into a pandas dataframe for easy readibility in terminal
	confusion_matrix = pd.DataFrame(confusion_matrix, index=true_actions, columns=pred_actions)

	return confusion_matrix