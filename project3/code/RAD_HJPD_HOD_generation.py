import pathlib
import math
import copy
import numpy as np
from numpy import linalg as LA
import pandas as pd
import os


def generate_representations(saved_rep_dir, output_paths_train, output_paths_test, rad_ang_bins, rad_dist_bins, hjpd_bins, hod_bins):
    """ Generates three feature representations (RAD, HJPD, and HOD) for each file contained in the train and test directory.
        These representations are output with each file being represented as a single line in each of their output files.

    Input:
        rad_ang_bins (int): the number of bins used in the RAD angle histogram
        rad_dist_bins (int): the number of bins used in the RAD distance histogram
        hjpd_bins (int): the number of bins used in the HJPD histogram
        hod_bins (int):  the number of bins used in the HOD histogram

    Output:
        None
    """
    # Defines input data file paths
    cwd = pathlib.Path.cwd()  # gets current working directory
    train_dir = cwd.joinpath('project3', 'dataset', 'train')  # Generates training data filepath
    test_dir = cwd.joinpath('project3', 'dataset', 'test')  # Generates training data filepath

    # Defines column names for data frames
    columns = ['frame_id', 'joint_id', 'joint-pos_x', 'joint-pos_y', 'joint-pos_z']

    # Defines directorie holding training & test data
    data_directories = [train_dir, test_dir]

    # Creates a dictionary of dictionaries to store the representations
    train_representations = {
        'rad': dict(),
        'hjpd': dict(),
        'hod': dict()
    }
    test_representations = {
        'rad': dict(),
        'hjpd': dict(),
        'hod': dict()
    }

    # Creates a dictionary of dictionaries to store the representations in the form of histograms
    histograms_train = {
        'rad': dict(),
        'hjpd': dict(),
        'hod': dict()
    }
    histograms_test = {
        'rad': dict(),
        'hjpd': dict(),
        'hod': dict()
    }

    # Loops over each file and generates representation for each file
    file_count = 0
    for data_directory in data_directories:
        print("")

        # Count number of files in directory to print progress to terminal
        path, dirs, files = next(os.walk(data_directory))
        num_files = len(files)
        if data_directory == train_dir:
            print("Processing Train Data Files")
            dir_file_count = 0
        else:
            print("Processing Test Data Files")
            dir_file_count = 0

        for data_file in data_directory.iterdir():
            # Convert .txt file of data into np array
            file_raw_data = np.loadtxt(data_file)

            # Convert np array into pandas DataFrame
            file_dataframe = pd.DataFrame(file_raw_data, columns=columns)  # label columns
            file_dataframe = file_dataframe.fillna(value=0)  # Ensure there are no N/A or NaN values in data

            # Generate RAD, HJPD, and HOD representations for training data
            if data_directory == train_dir:
                # RAD - Get 2D matrix of RAD parameter (angles & joint distances) as well as the mins and maxs
                train_representations['rad'][data_file.name], rad_file_min, rad_file_max = RAD(file_dataframe)

                # HJPD - Get 2D matrix of HJPD parameters ([x,y,z] distances) as well as the mins and maxs
                train_representations['hjpd'][data_file.name], hjpd_file_min, hjpd_file_max = HJPD(file_dataframe)

                # HOD - Get concatenated pyramid of histograms, which show occurrence of angles in principle plane projections
                train_representations['hod'][data_file.name] = HOD(file_dataframe, hod_bins)

            # Generate RAD, HJPD, and HOD representations for testing data
            else:
                # RAD - Get 2D matrix of RAD parameter (angles & joint distances)
                test_representations['rad'][data_file.name], _, _ = RAD(file_dataframe)

                # HJPD - Get 2D matrix of HJPD parameters ([x,y,z] distances)
                test_representations['hjpd'][data_file.name], _, _ = HJPD(file_dataframe)

                # HOD - Get concatenated pyramid of histograms, which show occurrence of angles in principle plane projections
                test_representations['hod'][data_file.name] = HOD(file_dataframe, hod_bins)

            # Create matrices of mins and maxs
            if file_count == 0:
                rad_mins = rad_file_min
                rad_maxs = rad_file_max
                hjpd_mins = hjpd_file_min
                hjpd_maxs = hjpd_file_max
            else:
                rad_mins = np.column_stack((rad_mins, rad_file_min))
                rad_maxs = np.column_stack((rad_maxs, rad_file_max))
                hjpd_mins = np.column_stack((hjpd_mins, hjpd_file_min))
                hjpd_maxs = np.column_stack((hjpd_maxs, hjpd_file_max))

            file_count += 1
            dir_file_count += 1

            print("File '{0}' Completed! Number of files processed: {1} / {2}.".format(data_file.name, dir_file_count, num_files))
    print("All Files Completed!")

    # Define the minimum and maximum ranges for RAD & HJPD histograms
    rad_min = rad_mins.min(axis=1)
    rad_max = rad_maxs.max(axis=1)
    hjpd_min = hjpd_mins.min(axis=1)
    hjpd_max = hjpd_maxs.max(axis=1)

    # Creates histogram for each representation
    print("")
    print("Creating Histograms")

    # Create histogram for RAD training data
    for file_name, rad_vals in train_representations['rad'].items():
        # Get number of parameter values (5 angles, 5 distances)
        num_angs = int(rad_vals.shape[0]) // 2
        num_dists = int(rad_vals.shape[0]) // 2

        # Initialize the histogram variables for the RAD angles
        rad_ang_hist = np.zeros((num_angs, rad_ang_bins))  # RAD angle histogram - not normalized
        rad_norm_angs = np.zeros(num_angs)  # Vector holding normalized RAD angle values for each skeleton frame
        rad_ang_hist_norm = np.zeros((num_angs, rad_ang_bins))  # RAD angle histogram - normalized

        # Generate normalized RAD histogram relative to space min and max of training data
        for ang in range(num_angs):
            # Generate non-normalized histogram for each angle type in all frames
            rad_ang_hist[ang:], _ = np.histogram(rad_vals[ang, :], bins=rad_ang_bins,
                                                 range=(rad_min[ang], rad_max[ang]))

            # Calculate number of skeleton frames the histogram is composed of
            rad_norm_angs[ang] = np.sum(rad_ang_hist[ang, :])

            # Normalize each bin angle occurrence value in histogram to be percentages of total number of frames
            rad_ang_hist_norm[ang, :] = [frame_ang / rad_norm_angs[ang] for frame_ang in rad_ang_hist[ang, :]]

        # Initialize the histogram variables for the RAD distances
        rad_dist_hist = np.zeros((num_dists, rad_dist_bins))  # RAD distance histogram - not normalized
        rad_norm_dists = np.zeros(num_dists)  # Vector holding normalized RAD distance values for each skeleton frame
        rad_dist_hist_norm = np.zeros((num_dists, rad_dist_bins))  # RAD distance histogram - normalized

        # Generate normalized RAD histogram relative to space min and max of training data
        for dist in range(num_dists):
            # Generate non-normalized histogram for each distance type in all frames
            rad_dist_hist[dist:], _ = np.histogram(rad_vals[dist + num_angs, :], bins=rad_dist_bins,
                                                   range=(rad_min[dist + num_angs], rad_max[dist + num_angs]))

            # Calculate number of skeleton frames the histogram is composed of
            rad_norm_dists[dist] = np.sum(rad_dist_hist[dist, :])

            # Normalize each bin distance occurrence value in histogram to be percentages of total number of frames
            rad_dist_hist_norm[dist, :] = [frame_dist / rad_norm_dists[dist] for frame_dist in rad_dist_hist[dist, :]]

        # Flatten histogram variables from 5 parameters x B bins matrix to be 5*B vector
        rad_ang_hist_norm = rad_ang_hist_norm.flatten()  # Histogram size is 5*B_ang, where B_ang is rad_ang_bins
        rad_dist_hist_norm = rad_dist_hist_norm.flatten()  # Histogram size: 5*B_dist, where B_dist is rad_dist_bins

        # Combine all histograms into single vector of histogram values. Histogram size is 5 * (B_ang + B_dist)
        rad_hist_norm = np.concatenate((rad_ang_hist_norm, rad_dist_hist_norm))

        # Assign to training histograms dictionary
        histograms_train['rad'][file_name] = rad_hist_norm

    # Create histogram for RAD testing data
    for file_name, rad_vals in test_representations['rad'].items():
        # Get number of parameter values (5 angles, 5 distances)
        num_angs = int(rad_vals.shape[0]) // 2
        num_dists = int(rad_vals.shape[0]) // 2

        # Initialize the histogram variables for the RAD angles
        rad_ang_hist = np.zeros((num_angs, rad_ang_bins))  # RAD angle histogram - not normalized
        rad_norm_angs = np.zeros(num_angs)  # Vector holding normalized RAD angle values for each skeleton frame
        rad_ang_hist_norm = np.zeros((num_angs, rad_ang_bins))  # RAD angle histogram - normalized

        # Generate normalized RAD histogram relative to space min and max of training data
        for ang in range(num_angs):
            # Generate non-normalized histogram for each angle type in all frames
            rad_ang_hist[ang:], _ = np.histogram(rad_vals[ang, :], bins=rad_ang_bins,
                                                 range=(rad_min[ang], rad_max[ang]))

            # Calculate number of skeleton frames the histogram is composed of
            rad_norm_angs[ang] = np.sum(rad_ang_hist[ang, :])

            # Normalize each bin angle occurrence value in histogram to be percentages of total number of frames
            rad_ang_hist_norm[ang, :] = [frame_ang / rad_norm_angs[ang] for frame_ang in rad_ang_hist[ang, :]]

        # Initialize the histogram variables for the RAD distances
        rad_dist_hist = np.zeros((num_dists, rad_dist_bins))  # RAD distance histogram - not normalized
        rad_norm_dists = np.zeros(num_dists)  # Vector holding normalized RAD distance values for each skeleton frame
        rad_dist_hist_norm = np.zeros((num_dists, rad_dist_bins))  # RAD distance histogram - normalized

        # Generate normalized RAD histogram relative to space min and max of training data
        for dist in range(num_dists):
            # Generate non-normalized histogram for each distance type in all frames
            rad_dist_hist[dist:], _ = np.histogram(rad_vals[dist + num_angs, :], bins=rad_dist_bins,
                                                   range=(rad_min[dist + num_angs], rad_max[dist + num_angs]))

            # Calculate number of skeleton frames the histogram is composed of
            rad_norm_dists[dist] = np.sum(rad_dist_hist[dist, :])

            # Normalize each bin distance occurrence value in histogram to be percentages of total number of frames
            rad_dist_hist_norm[dist, :] = [frame_dist / rad_norm_dists[dist] for frame_dist in rad_dist_hist[dist, :]]

        # Flatten histogram variables from 5 parameters x B bins matrix to be 5*B vector
        rad_ang_hist_norm = rad_ang_hist_norm.flatten()  # Histogram size is 5*B_ang, where B_ang is rad_ang_bins
        rad_dist_hist_norm = rad_dist_hist_norm.flatten()  # Histogram size: 5*B_dist, where B_dist is rad_dist_bins

        # Combine all histograms into single vector of histogram values. Histogram size is 5 * (B_ang + B_dist)
        rad_hist_norm = np.concatenate((rad_ang_hist_norm, rad_dist_hist_norm))

        # Assign to testing histograms dictionary
        histograms_test['rad'][file_name] = rad_hist_norm

    print("")
    print("RAD Training & Testing Histograms Completed")

    # Create histogram for HJPD training data
    for file_name, hjpd_vals in train_representations['hjpd'].items():
        # Get number of distance values
        num_dists = int(hjpd_vals.shape[0])
        num_features = int(num_dists / 3)

        # Initialize the histogram variables for the RAD angles
        hjpd_hist = np.zeros((num_dists, hjpd_bins))  # HJPD histogram - not normalized & [x,y,z] features separated
        hjpd_norm = np.zeros(num_dists)  # Vector holding normalized HJPD values for each skeleton frame
        hjpd_hist_norm = np.zeros((num_dists, hjpd_bins))  # HJPD histogram - normalized & [x,y,z] features separated
        hjpd_hist_norm_comb = np.zeros((num_features, hjpd_bins*3))  # HJPD histogram - normalized & [x,y,z] features combined

        # Generate normalized HJPD histogram relative to space min and max of training data
        feature = 0  # Count of [x,y,z] displacement combinations
        for dist in range(num_dists):
            # Generate non-normalized histogram for each distance type in all frames
            hjpd_hist[dist:], _ = np.histogram(hjpd_vals[dist, :], bins=hjpd_bins,
                                               range=(hjpd_min[dist], hjpd_max[dist]))

            # Calculate number of skeleton frames the histogram is composed of
            hjpd_norm[dist] = np.sum(hjpd_hist[dist, :])

            # Normalize each bin distance occurrence value in histogram to be percentages of total number of frames
            hjpd_hist_norm[dist, :] = [x / hjpd_norm[dist] for x in hjpd_hist[dist, :]]

            # Combine all histograms into single vector of histogram values. Histogram size is (57 / 3) * (B * 3)
            if (dist+1) % 3 == 0:
                hjpd_hist_norm_comb[feature, :] = np.concatenate((hjpd_hist_norm[dist-2, :],  # x displacement vectors
                                                                  hjpd_hist_norm[dist-1, :],  # y displacement vectors
                                                                  hjpd_hist_norm[dist, :]))   # z displacement vectors
                feature += 1

        # Assign to training histograms dictionary
        histograms_train['hjpd'][file_name] = hjpd_hist_norm_comb

    # Create histogram for HJPD testing data
    for file_name, hjpd_vals in test_representations['hjpd'].items():
        # Get number of distance values
        num_dists = int(hjpd_vals.shape[0])
        num_features = int(num_dists / 3)

        # Initialize the histogram variables for the RAD angles
        hjpd_hist = np.zeros((num_dists, hjpd_bins))  # HJPD histogram - not normalized & [x,y,z] features separated
        hjpd_norm = np.zeros(num_dists)  # Vector holding normalized HJPD values for each skeleton frame
        hjpd_hist_norm = np.zeros((num_dists, hjpd_bins))  # HJPD histogram - normalized & [x,y,z] features separated
        hjpd_hist_norm_comb = np.zeros((num_features, hjpd_bins*3))  # HJPD histogram - normalized & [x,y,z] features combined

        # Generate normalized HJPD histogram relative to space min and max of training data
        feature = 0  # Count of [x,y,z] displacement combinations
        for dist in range(num_dists):
            # Generate non-normalized histogram for each distance type in all frames
            hjpd_hist[dist:], _ = np.histogram(hjpd_vals[dist, :], bins=hjpd_bins,
                                               range=(hjpd_min[dist], hjpd_max[dist]))

            # Calculate number of skeleton frames the histogram is composed of
            hjpd_norm[dist] = np.sum(hjpd_hist[dist, :])

            # Normalize each bin distance occurrence value in histogram to be percentages of total number of frames
            hjpd_hist_norm[dist, :] = [x / hjpd_norm[dist] for x in hjpd_hist[dist, :]]

            # Combine all histograms into single vector of histogram values.
            # Each histogram size is (57 / 3) * (B * 3).
            # Because we are comparing 19 joints to the center joint, there will be 19 histograms per data file.
            if (dist+1) % 3 == 0:
                hjpd_hist_norm_comb[feature, :] = np.concatenate((hjpd_hist_norm[dist-2, :],  # x displacement vectors
                                                                  hjpd_hist_norm[dist-1, :],  # y displacement vectors
                                                                  hjpd_hist_norm[dist, :]))   # z displacement vectors
                feature += 1

        histograms_test['hjpd'][file_name] = hjpd_hist_norm_comb

    print("")
    print("HJPD Training & Testing Histograms Completed")

    # Copy both the training & testing histograms for HOD into the train & test histogram dictionaries
    histograms_train['hod'] = copy.deepcopy(train_representations['hod'])
    histograms_test['hod'] = copy.deepcopy(test_representations['hod'])

    print("")
    print("HOD Training & Testing Histograms Completed")

    # Save histograms to file
    print("")
    print("Saving to files.")
    write_data_to_file(histograms_train, output_paths_train)
    write_data_to_file(histograms_test, output_paths_test)
    print("")


def write_data_to_file(dictionary, file_save_paths):
    """ Writes a dictionary of dictionaries into a libsvm compatible format with the label being extracted from the
        file name.

    Input:
        dict (dictionary): a dictionary of dictionaries where the outer keys correspond to the representation name
                            and the inner keys correspond to a file name
        file_save_paths (np.ndarray): a list of file paths generated using libpath to be written to

    Output:
        None

    """
    for algorithm_name, algorithm_data in dictionary.items():
        # Initialize list holding all data for a given histogram.
        histogram_data_list = []

        # Save HJPD histograms
        # As stated at line 306, there are 19 different histograms per data file instead of one, like RAD and HOD.
        # Therefore, an additional for loop is required for looping through data.
        if algorithm_name == 'hjpd':
            for filename, histogram_data in algorithm_data.items():
                # Get index of activity number within file name. For example, if file name is
                # 'a10_s06_e02_skeleton_proj.txt', the activity is 'a10' and the activity number is 10.
                activity_num = [filename[1:3]]

                # Cycle through each histogram bin value and save value to file
                bin_val_index = 1  # define index of bin value within histogram
                for histogram in histogram_data:
                    for bin_val in histogram:
                        # Generate list of bin values with corresponding index within histogram attached to activity number
                        activity_num.append('{0}:{1}'.format(bin_val_index, bin_val))
                        bin_val_index += 1  # increase bin index value

                histogram_data_list.append(' '.join(activity_num))  # Convert from size of list to be of size 1

            file_string = '\n'.join(histogram_data_list)  # Convert to string
            file_save_paths[1].write_text(file_string)  # Write string to file

        # Save RAD & HOD histograms
        else:
            for filename, histogram_data in algorithm_data.items():
                # Get index of activity number within file name. For example, if file name is
                # 'a10_s06_e02_skeleton_proj.txt', the activity is 'a10' and the activity number is 10.
                activity_num = [filename[1:3]]

                # Cycle through each histogram bin value and save value to file
                bin_val_index = 1  # define index of bin value within histogram
                for bin_val in histogram_data:
                    # Generate list of bin values with corresponding index within histogram attached to activity number
                    activity_num.append('{0}:{1}'.format(bin_val_index, bin_val))
                    bin_val_index += 1  # increase bin index value

                histogram_data_list.append(' '.join(activity_num))  # Convert from size of list to be of size 1
            file_string = '\n'.join(histogram_data_list)    # Convert to string

            if algorithm_name == 'rad':
                file_save_paths[0].write_text(file_string)  # Write string to file
            else:
                file_save_paths[2].write_text(file_string)  # Write string to file


def RAD(data):
    """ Calculates the Relative Angle and Distance (RAD) representation of a star skeleton

    Input:
        data (pd.DataFrame): an Nx5 matrix describing 6 joint positions at a given instant where N is the number of
                             frames in the given dataset.
                             ['frame_id', 'joint_id', 'joint-pos_x', 'joint-pos_y', 'joint-pos_z']
                             The joint_id correspondence is as follows:
                                   joint_id = 1 -> Body Center
                                   joint_id = 4 -> Head
                                   joint_id = 8 -> Right Hand
                                   joint_id = 12 -> Left Hand
                                   joint_id = 16 -> Right Foot
                                   joint_id = 20 -> Left Foot

    Ouput:
        rad_vals (np.ndarray): a 10xN matrix describing the relative angle and distance between joints where N is the
                               number of frames in the given dataset.
                               Each column follows the following structure for N frames
                                   a1 - Angle between Head and Right Hand (4 to 8)
                                   a2 - Angle between Head and Left Hand (4 to 12)
                                   a3 - Angle between Right Hand and Right Foot (8 to 16)
                                   a4 - Angle between Left Hand and Left Foot (12 to 20)
                                   a5 - Angle between Right Foot and Left Foot (16 to 20)
                                   d1 - Distance from Body Center to Head (1 to 4)
                                   d2 - Distance from Body Center to Right Hand (1 to 8)
                                   d3 - Distance from Body Center to Left Hand (1 to 12)
                                   d4 - Distance from Body Center to Right Foot (1 to 16)
                                   d5 - Distance from Body Center to Left Foot (1 to 20)

          min_vals (np.ndarray): a 10x1 vector describing the minimum of each angle & distance for each angle &
                                 distance type in all frames

          max_vals (np.ndarray): a 10x1 vector describing the maximum of each angle & distance for each angle &
                                 distance type in all frames

    """
    # Create data frame only containing relevant joints
    data = data[data['joint_id'].isin([1, 4, 8, 12, 16, 20])]

    # Initialize matrix that will hold values for each star skeleton's relative angles and distance between joints
    # Matrix will be of size 10xN, where N is the number of star skeletons
    num_frames = int(data['frame_id'].max())  # Number of star skeletons, or frames in data
    rad_vals = np.zeros([10, num_frames])

    for frame in range(num_frames):
        # Obtain current frame number
        current_frame = frame + 1

        # Extracts all data from current frame as a numpy array
        frame_data = data[data['frame_id'].isin([current_frame])]  # Data frame for current frame
        frame_data = frame_data.values  # Numpy array of data points

        # Distance vectors between skeleton star points & skeleton center
        dist_vectors = np.zeros([5, 3])

        for joint in range(5):
            # Calculates [x,y,z] vector from Body Center to each extremity
            # joint(x,y,z) = frame_data[joint, 2:]
            dist_vectors[joint, :] = frame_data[joint+1, 2:] - frame_data[0, 2:]

        # Initialize vector holding angles between distance vectors held in variable "dist_vectors"
        rad_angs = np.zeros([5, 1])

        # Calculate the angle between each vector connecting the skeleton star point to the skeleton center
        rad_angs[0] = calc_angle(dist_vectors[0, :], dist_vectors[1, :])  # angle between head & right hand
        rad_angs[1] = calc_angle(dist_vectors[0, :], dist_vectors[2, :])  # angle between head & left hand
        rad_angs[2] = calc_angle(dist_vectors[1, :], dist_vectors[3, :])  # angle between right hand & right foot
        rad_angs[3] = calc_angle(dist_vectors[2, :], dist_vectors[4, :])  # angle between left hand & left foot
        rad_angs[4] = calc_angle(dist_vectors[3, :], dist_vectors[4, :])  # angle between right foot & left foot

        # Initialize vector holding euclidean distances between skeleton star points & skeleton center
        rad_dist = np.zeros([5, 1])

        # Calculate the euclidean distance between skeleton star points and skeleton center
        for star_point in range(5):
            rad_dist[star_point] = np.sqrt(np.sum((frame_data[0, 2:] - frame_data[star_point + 1, 2:]) ** 2, axis=0))

        # Save rad_angs and rad_dist to rad_vals
        rad_vals[0:5, frame] = rad_angs[:, 0]
        rad_vals[5:10, frame] = rad_dist[:, 0]

    # Extract the maximum and minimum angle & distance for each angle & distance type in all frames.
    # These will be used later when making histograms for defining range in data.
    min_vals = rad_vals.min(axis=1)
    max_vals = rad_vals.max(axis=1)

    return rad_vals, min_vals, max_vals


def HJPD(data):
    """ Calculates the Histogram of Joint Position Differences (HJPD)

    Input:
        data (np.ndarray): an Nx5 matrix describing 6 joint positions at a given instant where N is the number of
                           frames in the given dataset. Each frame has 20 joints.
                           ['frame_id', 'joint_id', 'joint-pos_x', 'joint-pos_y', 'joint-pos_z']

    Output:
        hjpd_vals (np.ndarray): a 57xN matrix describing the relative displacement of all joints
                                with reference to joint_id = 1 (hip center) in [x,y,z]
                                 delta_x(1,2) = x2 - x1
                                 delta_y(1,2) = y2 - y1
                                 delta_z(1,2) = z2 - z1
                                 delta_x(1,3) = x3 - x1
                                 ...
                                 delta_z(1,20) = z20 - z1

        min_vals (np.ndarray): a 10x1 vector describing the minimum of each distance vector value for each
                               joint-to-joint type distance in all frames

        max_vals (np.ndarray): a 10x1 vector describing the maximum of each distance vector value for each
                               joint-to-joint type distance in all frames
    """
    num_frames = int(data['frame_id'].max())  # Number of frames capturing skeleton joints in data
    num_joints = 20  # Number of joints in frame
    hjpd_vals = np.zeros([3*(num_joints-1), num_frames])  # Initialize the matrix containing all HJPD representations

    for frame in range(num_frames):
        # Obtain current frame number
        current_frame = frame + 1

        # Extracts all data from current frame as a numpy array
        frame_data = data[data['frame_id'].isin([current_frame])]  # Data frame for current frame
        frame_data = frame_data.values  # Numpy array of data points

        # Initialize vector holding vectors between joint points & skeleton center
        dist_vectors = np.zeros([num_joints-1, 3])

        # Calculate the HJPD representation of the frame and save to dist_vectors
        for joint in range(num_joints-1):
            # Calculates [x,y,z] vector from joint_id = 1 to each other joint
            # joint(x,y,z) = frame_data[joint, 2:]
            dist_vectors[joint, :] = frame_data[joint + 1, 2:] - frame_data[0, 2:]

        # Initialize the vector holding all HJPD representations for the current frame
        hjpd_reps_current_frame = np.zeros([3 * (num_joints - 1), 1])

        # Extracts [x,y,z] vector values from vectors and makes into a list
        for joint in range(num_joints-1):
            hjpd_reps_current_frame[3 * joint] = dist_vectors[joint, 0]  # Extract x value
            hjpd_reps_current_frame[(3 * joint) + 1] = dist_vectors[joint, 1]  # Extract y value
            hjpd_reps_current_frame[(3 * joint) + 2] = dist_vectors[joint, 2]  # Extract z value

        # Save all vector values for each frame
        hjpd_vals[:, frame] = hjpd_reps_current_frame[:, 0]

    # Extract the maximum and minimum distance vector values for each joint-to-joint type distance in all frames.
    # These will be used later when making histograms for defining range in data.
    min_vals = hjpd_vals.min(axis=1)
    max_vals = hjpd_vals.max(axis=1)

    return hjpd_vals, min_vals, max_vals


def HOD(data, bins):
    """ Calculates the Histogram of Joint Position Differences (HOD)

    Input:
        data (pd.DataFrame): an Nx5 matrix describing 6 joint positions at a given instant where N is the number of
                             frames in the given dataset. Each frame has 20 joints.
                             ['frame_id', 'joint_id', 'joint-pos_x', 'joint-pos_y', 'joint-pos_z']

        bins (int): Number of bins in HOD histogram.

    Output:
        hod_rep (np.ndarray): A single line array that describes a 3 level temporal pyramid HOD representation.
    """
    num_frames = int(data['frame_id'].max())  # Number of frames capturing skeleton joints in data

    # Initialize vector holding angles in principle plane projections between joint i and joint i+1.
    # There are three angles per joint:
    # (1) Angle between XY projection
    # (2) Angle between YZ projection
    # (3) Angle between XZ projection
    hod_vals = np.zeros([60, num_frames])

    # Calculate angles for each frame
    for frame in range(num_frames-1):
        # Obtain current frame number
        current_frame = frame + 1

        # Extract data from first frame for first iteration.
        current_frame_data = data[data['frame_id'].isin([current_frame])]
        current_frame_data = current_frame_data.values

        # Extracts data from next frame. Current frame will become the next frame for all iterations except first
        next_frame_data = data[data['frame_id'].isin([current_frame + 1])]
        next_frame_data = next_frame_data.values
        # print("Data = ", current_frame_data)

        # Calculate distance vector from current frame joints to next frame joints
        # joint(x,y,z) = frame_data[joint, 2:]
        dist_vectors = next_frame_data[:, 2:] - current_frame_data[:, 2:]

        # Initialize vector to store calculated angles for current frame
        current_frame_angs = np.zeros(60)

        # Calculate angles in principle plane projections and store in a vector of angles for each joint.
        for joint in range(20):
            current_frame_angs[3 * joint] = math.atan2(dist_vectors[joint, 1], dist_vectors[joint, 0])  # xy angle
            current_frame_angs[3 * joint + 1] = math.atan2(dist_vectors[joint, 2], dist_vectors[joint, 1])  # yz angle
            current_frame_angs[3 * joint + 2] = math.atan2(dist_vectors[joint, 2], dist_vectors[joint, 0])  # xz angle
            # print("Cur Frame: {}, Next Frame: {}, frame angle: {}, {}, {}".format(current_frame, current_frame + 1, current_frame_angs[3 * joint], current_frame_angs[3 * joint+1], current_frame_angs[3 * joint+2]))

        # Save angles to matrix of holding angles for all frames.
        hod_vals[:, frame] = current_frame_angs

    # Determine range of skeleton frames for how much half and quarter histograms will hold
    half_frames = hod_vals.shape[1] // 2  # Half of skeleton frames
    quarter_frames = hod_vals.shape[1] // 4  # Quarter of skeleton frames

    # Initialize histogram arrays
    hod_hist = np.zeros([60, bins])  # Histogram containing the entire data set
    hod_hist_h1 = np.zeros([60, bins])  # Histogram containing the 1st half of the data set
    hod_hist_h2 = np.zeros([60, bins])  # Histogram containing the 2nd half of the data set
    hod_hist_q1 = np.zeros([60, bins])  # Histogram containing the 1st quarter of the data set
    hod_hist_q2 = np.zeros([60, bins])  # Histogram containing the 2nd quarter of the data set
    hod_hist_q3 = np.zeros([60, bins])  # Histogram containing the 3rd quarter of the data set
    hod_hist_q4 = np.zeros([60, bins])  # Histogram containing the 4th quarter of the data set

    # Create histogram using all HOD angles for all skeleton frames.
    # Histograms show number of angles within bin angle range that show up within specified frame range.
    for ang in range(60):
        # Generates a histogram with specified number of bins for each displacement on range (-pi,pi)
        hod_hist[ang, :], _ = np.histogram(hod_vals[ang, :], bins, range=(-math.pi, math.pi))
        hod_hist_h1[ang:], _ = np.histogram(hod_vals[ang, :half_frames], bins, range=(-math.pi, math.pi))
        hod_hist_h2[ang:], _ = np.histogram(hod_vals[ang, half_frames:], bins, range=(-math.pi, math.pi))
        hod_hist_q1[ang:], _ = np.histogram(hod_vals[ang, :quarter_frames], bins, range=(-math.pi, math.pi))
        hod_hist_q2[ang:], _ = np.histogram(hod_vals[ang, quarter_frames:half_frames], bins, range=(-math.pi, math.pi))
        hod_hist_q3[ang:], _ = np.histogram(hod_vals[ang, half_frames:(3 * quarter_frames):], bins, range=(-math.pi, math.pi))
        hod_hist_q4[ang:], _ = np.histogram(hod_vals[ang, (3 * quarter_frames):], bins, range=(-math.pi, math.pi))

        # Calculate number of skeleton frames each histogram is composed of
        hod_hist_norm = np.sum(hod_hist[ang, :])
        hod_hist_norm_h1 = np.sum(hod_hist_h1[ang, :])
        hod_hist_norm_h2 = np.sum(hod_hist_h2[ang, :])
        hod_hist_norm_q1 = np.sum(hod_hist_q1[ang, :])
        hod_hist_norm_q2 = np.sum(hod_hist_q2[ang, :])
        hod_hist_norm_q3 = np.sum(hod_hist_q3[ang, :])
        hod_hist_norm_q4 = np.sum(hod_hist_q4[ang, :])

        # Normalize each bin angle occurrence value in histogram to be percentages of total number of frames
        hod_hist[ang, :] = [frame_ang / hod_hist_norm for frame_ang in hod_hist[ang, :]]
        hod_hist_h1[ang:] = [frame_ang / hod_hist_norm_h1 for frame_ang in hod_hist_h1[ang, :]]
        hod_hist_h2[ang:] = [frame_ang / hod_hist_norm_h2 for frame_ang in hod_hist_h2[ang, :]]
        hod_hist_q1[ang:] = [frame_ang / hod_hist_norm_q1 for frame_ang in hod_hist_q1[ang, :]]
        hod_hist_q2[ang:] = [frame_ang / hod_hist_norm_q2 for frame_ang in hod_hist_q2[ang, :]]
        hod_hist_q3[ang:] = [frame_ang / hod_hist_norm_q3 for frame_ang in hod_hist_q3[ang, :]]
        hod_hist_q4[ang:] = [frame_ang / hod_hist_norm_q4 for frame_ang in hod_hist_q4[ang, :]]

    # Flatten histogram variables from 60 angles x B bins matrix to be 60*B vector
    hod_hist = hod_hist.flatten()
    hod_hist_h1 = hod_hist_h1.flatten()
    hod_hist_h2 = hod_hist_h2.flatten()
    hod_hist_q1 = hod_hist_q1.flatten()
    hod_hist_q2 = hod_hist_q2.flatten()
    hod_hist_q3 = hod_hist_q3.flatten()
    hod_hist_q4 = hod_hist_q4.flatten()

    # Combine all histograms into single vector of histogram values
    hod_rep = np.concatenate((hod_hist, hod_hist_h1, hod_hist_h2, hod_hist_q1, hod_hist_q2, hod_hist_q3, hod_hist_q4))

    return hod_rep


def calc_angle(vector1, vector2):
    """ Calculates the angle between two vectors using a permutation of law of cosines

    Input:
        vector1 (np.ndarray): an nx1 vector describing the vector between two points
        vector2 (np.ndarray): an nx1 vector describing the vector between two points

    Output:
        theta (float): a single radian value that describes the angle between two vectors

    """
    # Check if there are any distance values equal to zero in the distance vectors.
    # Each variable true_distance_vec# is size of vector input containing boolean
    # if zero is present (true) or not (false).
    true_distance_vec1 = np.isin(vector1, 0)
    true_distance_vec2 = np.isin(vector2, 0)

    # Check if all distance vector values are zero or not. If zero, there is no real distance vector since points
    # are the same. As a result, set the angle to be zero.
    if np.all(true_distance_vec1) or np.all(true_distance_vec2):
        # If zero, there is no real distance vector since points are the same. As a result, set the angle to be zero.
        angle = 0
    else:
        # If not zero, these are both true distance vectors. Calculate the angle between them.
        angle = np.arccos(np.dot(vector1, vector2) / np.dot((LA.norm(vector1)), (LA.norm(vector2))))

    return angle