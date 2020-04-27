import pathlib
import sys
import argparse
import os
import numpy as np
import pandas as pd

# Import custom functions
import project3.code as p3

cwd = pathlib.Path.cwd()

def main():
    print("")
    # Get list of user input arguments
    args = get_args()

    # Clears target directory and then creates representation data files if specified
    if args.new:
        print("**CREATING NEW REPRESENTATIONS**")        
        if args.rad_ang_bins:
            rad_ang_bins = p3.rad_ang_bin_qty.interface()
        else:
            rad_ang_bins = p3.rad_ang_bin_qty.initialize_with_default_values()
        
        if args.rad_dist_bins:
            rad_dist_bins = p3.rad_dist_bin_qty.interface()
        else:
            rad_dist_bins = p3.rad_dist_bin_qty.initialize_with_default_values()

        if args.hjpd_bins:
            hjpd_bins = p3.hjpd_bin_qty.interface()
        else:
            hjpd_bins = p3.hjpd_bin_qty.initialize_with_default_values()

        if args.hod_bins:
            hod_bins = p3.hod_bin_qty.interface()
        else:
            hod_bins = p3.hod_bin_qty.initialize_with_default_values()
        
        print("Selected histogram bin sizes:")
        print("1) RAD Angle Bin Quantity = {}".format(rad_ang_bins))
        print("2) RAD Distance Bin Quantity = {}".format(rad_dist_bins))
        print("3) HJPD Bin Quantity = {}".format(hjpd_bins))
        print("4) HOD Bin Quantity = {}".format(hod_bins))

        # Define directory for saving the RAD, HJPD, and HOD representations
        save_folder_name = "ra{0}_rd{1}_hj{2}_hd{3}".format(rad_ang_bins, rad_dist_bins, hjpd_bins, hod_bins)
        saved_rep_dir = cwd.joinpath('project3', 'output_data', save_folder_name)

        # Define output data file paths
        rad_d2_path = saved_rep_dir.joinpath('rad_d2')  # Generates output file path for rad_d2
        rad_d2t_path = saved_rep_dir.joinpath('rad_d2.t')  # Generates output file path for rad_d2.t
        hjpd_d2_path = saved_rep_dir.joinpath('hjpd_d2')  # Generates output file path for hjpd_d2
        hjpd_d2t_path = saved_rep_dir.joinpath('hjpd_d2.t')  # Generates output file path for hjpd_d2.t
        hod_d2_path = saved_rep_dir.joinpath('hod_d2')  # Generates output file path for hod_d2
        hod_d2t_path = saved_rep_dir.joinpath('hod_d2.t')  # Generates output file path for hod_d2.t

        # Defines directories where representation data files are saved
        output_paths_train = [rad_d2_path, hjpd_d2_path, hod_d2_path]
        output_paths_test = [rad_d2t_path, hjpd_d2t_path, hod_d2t_path]

        # If directory already exists, delete files within directory so they can be replaced
        if saved_rep_dir.exists():
            for file in saved_rep_dir.iterdir():
                file.unlink()
        # Otherwise, create directory
        else:
            saved_rep_dir.mkdir()

        # Generate the RAD, HJPD, and HOD representations. Once generated, save to files within saved_rep_dir
        p3.RAD_HJPD_HOD_generation.generate_representations(saved_rep_dir, output_paths_train, output_paths_test,
                                                            rad_ang_bins, rad_dist_bins, hjpd_bins, hod_bins)

        # Perform grid search on training representations
        print("RAD training representation grid search:")
        p3.train_test_model.grid_search(save_folder_name, 'rad_d2')

        print("HJPD training representation grid search:")
        p3.train_test_model.grid_search(save_folder_name, 'hjpd_d2')

        print("HOD training representation grid search:")
        p3.train_test_model.grid_search(save_folder_name, 'hod_d2')

        print("Grid Search Completed!")
        print("")
        print("All Representation files & Grid Search files saved! All saved files can be found at:")
        print(saved_rep_dir)
        print("")

    if args.train:
        print("**TRAINING MODEL**")
        if args.rad_c:
            rad_c = p3.rad_c_val.interface()
        else:
            rad_c = p3.rad_c_val.initialize_with_default_values()

        if args.rad_gamma:
            rad_gamma = p3.rad_gamma_val.interface()
        else:
            rad_gamma = p3.rad_gamma_val.initialize_with_default_values()

        if args.hjpd_c:
            hjpd_c = p3.hjpd_c_val.interface()
        else:
            hjpd_c = p3.hjpd_c_val.initialize_with_default_values()

        if args.hjpd_gamma:
            hjpd_gamma = p3.hjpd_gamma_val.interface()
        else:
            hjpd_gamma = p3.hjpd_gamma_val.initialize_with_default_values()

        if args.hod_c:
            hod_c = p3.hod_c_val.interface()
        else:
            hod_c = p3.hod_c_val.initialize_with_default_values()

        if args.hod_gamma:
            hod_gamma = p3.hod_gamma_val.interface()
        else:
            hod_gamma = p3.hod_gamma_val.initialize_with_default_values()

        print("Selected Hyperparameter Values:")
        print("1) RAD C value = {}".format(rad_c))
        print("2) RAD Gamma = {}".format(rad_gamma))
        print("3) HJPD C value = {}".format(hjpd_c))
        print("4) HJPD Gamma = {}".format(hjpd_gamma))
        print("5) HOD C value = {}".format(hod_c))
        print("6) HOD Gamma = {}".format(hod_gamma))
        print("")

        # Perform training on RAD data with defined parameters
        print('RAD - Predictions and Confusion Matrix')
        p3.train_test_model.train_test_model(rad_d2_path, rad_d2t_path, rad_c, rad_gamma)

        # Perform training on HJPD data with defined parameters
        print('HJPD - Predictions and Confusion Matrix')
        p3.train_test_model.train_test_model(hjpd_d2_path, hjpd_d2t_path, hjpd_c, hjpd_gamma)

        # Perform training on HOD data with defined parameters
        print('HOD - Predictions and Confusion Matrix')
        p3.train_test_model.train_test_model(hod_d2_path, hod_d2t_path, hod_c, hod_gamma)


def get_args():
    parser = argparse.ArgumentParser()

    # Get the command line arguments
    parser.add_argument('-n',
                        '--new',
                        action='store_true',
                        default=False,
                        help='Generates new representations from raw data.')
    parser.add_argument('-t',
                        '--train',
                        action='store_true',
                        default=False,
                        help='Trains RAD, HJPD, & HOD models using RAD, HJPD, & HOD representations.')
    parser.add_argument('--rad-ang-bins',
                        action='store_true',
                        default=False,
                        help='Manually set the bin number for the RAD angle histogram. Default bin size is 11.')
    parser.add_argument('--rad-dist-bins',
                        action='store_true',
                        default=False,
                        help='Manually set the bin number for the RAD distance histogram. Default bin size is 11.')
    parser.add_argument('--hjpd-bins',
                        action='store_true',
                        default=False,
                        help='Manually set the bin number for the HJPD histogram. Default bin size is 11.')
    parser.add_argument('--hod-bins',
                        action='store_true',
                        default=False,
                        help='Manually set the bin number for the HOD histogram. Default bin size is 22.')
    parser.add_argument('--rad-c',
                        action='store_true',
                        default=False,
                        help='Sets the C value for the RAD SVM training. Default C value is 8.0.')
    parser.add_argument('--rad-gamma',
                        action='store_true',
                        default=False,
                        help='Sets the gamma value for the RAD SVM training. Default gamma value is 0.03125.')
    parser.add_argument('--hjpd-c',
                        action='store_true',
                        default=False,
                        help='Sets the C value for the HJPD SVM training. Default C valueis 2.0.')
    parser.add_argument('--hjpd-gamma',
                        action='store_true',
                        default=False,
                        help='Sets the gamma value for the HJPD SVM training. Default gamma value is 0.125.')
    parser.add_argument('--hod-c',
                        action='store_true',
                        default=False,
                        help='Sets the C value for the HOD SVM training. Default C value is 2.0.')
    parser.add_argument('--hod-gamma',
                        action='store_true',
                        default=False,
                        help='Sets the gamma value for the HOD SVM training. Default gamma value 0.03125.')

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    sys.exit(main())
