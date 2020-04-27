<h1>Human Centered Robotics: Human Behavior Classifier</h1>

**Name:** Casey Duncan <br />
**CWID:** 10834922 <br />
**Date:** 04/27/2020 <br />
**Course:** CSCI 573 - Human Centered Robotics <br />
**Assignment:** Robot Understanding of Human Behaviors Using Skeleton-Based Representations <br />
**Task 2:** Robot Learning From Data (see line 48 for task 2 deliverables) <br />


<h2>OVERVIEW</h2>

The goal of this project was to use the skeleton data from the MSR Daily Activity 3D dataset to construct three skeleton-based human representations and then apply Support Vector Machines (SVMs) to enable robot behavior understanding using these representations on the LIBSVM library. These three representations are provided below:

1. Relative Angles and Distances (RAD)
http://inside.mines.edu/~hzhang/Courses/CSCI473-573/Projects/Project-3-Spring20-D1.pdf
2. Histogram of Joint Position Differences (HJPD)
https://staffhome.ecm.uwa.edu.au/~00053650/papers/hossein_WACV2014.pdf
3. Histogram of Oriented Displacements (HOD)
https://www.aaai.org/ocs/index.php/IJCAI/IJCAI13/paper/view/6967

The skeleton data is made of skeletal joint position data of people performing the following six activities:
1. Cheer Up (activity 08)
2. Toss Paper (activity 10)
3. Lie On Sofa (activity 12)
4. Walk (activity 13)
5. Stan dUp (activity 15)
6. Sit Down (activity 16)

The (x,y,z) positions of the following 20 skeletal joints are recorded for a variety of number of video frames to capture each activity:
1. Hip Center
2. Spine
3. Shoulder Center
4. Head
5. Shoulder Right
6. Elbow Right
7. Wrist Right
8. Hand Right
9. Shoulder Left
10. Elbow Left
11. Wrist Left
12. Hand Left
13. Hip Right
14. Knee Right
15. Ankle Right
16. Foot Right
17. Hip Left
18. Knee Left
19. Ankle Left
20. Foot Left

Descriptions of how I implemented each of the three representations using the skeletal joint position data is detailed in the **Solution Description** section.


<h2>TASK 2 DELIVERABLES</h2>

Listed below are the directory locations for the Task 2 required deliverables. More information on the directory layout within the *D2_casey_duncan* directory is provided in the **Directory Layout** section.

1. Grid Search graphs:
    - Directory: *D2_casey_duncan/project3/output_data/output_data_folder* <br />
    **NOTE:** See the layout for the *D2_casey_duncan/project3/output_data* directory in the **Directory Layout** section below for how the naming convention for each *output_data_folder* is formatted.

2. Best bin sizes, C values, & gamma values:
	- RAD: angle bins = 11, distance bins = 11, C = 8.0, gamma = 0.03125 --> Accuracy = 60.4167%
	- HJPD: bins = 11, C = 2.0, gamma = 0.125 --> Accuracy = 83.3333%
	- HOD: bins = 22, C = 2.0, gamma = 0.03125 --> Accuracy = 89.5833%

3. Converted Representation Files & Output Prediction Files:
	- Directory: *D2_casey_duncan/project3/output_data/output_data_folder* <br />
	**NOTE:** See the layout for the *D2_casey_duncan/project3/output_data* directory in the **Directory Layout** section below for how the naming convention for each *output_data_folder* is formatted.

4. Figure showing representation bin size vs. accuracy:
	- Directory: *D2_casey_duncan/project3/data_analysis*
	- File name: Bin Size vs Accuracy Plot.png

5. Python Code:
	- Directory: *D2_casey_duncan/project3/code* <br />
	**NOTE:** This contains all code used by main.py, which is within the *D2_casey_duncan* folder.


<h2>DIRECTORY LAYOUT</h2>
Within this directory are the following files:

1. *main.py* <br />
This file is used to run all of the code for project 3. See the **How to run code** section below for how to run these programs.

2. *poetry.lock* and *pyproject.toml* <br />
These files are used for installing the poetry python virtual environment containing all python packages required to run all of the code. If you do not have poetry installed on your computer, you can use your own python environment for running the code. However, you will need to install the packages shown in step 2 of the **How to run code** section below.

There are also the following folders within this directory:

1. *D2_casey_duncan/project3* <br />
This containes the folders *code*, *dataset*, *data_analysis*, and *output_data*.

2. *D2_casey_duncan/project3/code* <br />
This contains all code used by main.py.

3. *D2_casey_duncan/project3/dataset* <br />
This contains all of the training & test data.

4. *D2_casey_duncan/project3/data_analysis* <br />
This contains an excel file comparing the representation bin sizes and results grid search accuracies, as well as an image of this data plotted.

5. *D2_casey_duncan/project3/output_data* <br />
This contains several folders holding all of the saved representations training & testing data. Each folder name represents the number of bins used when generating each representation. For example, folder *ra10_rd10_hj24_hd14* holds the data for:

    - RAD representation trained & tested using 10 bins for RAD angles (*ra*) and 10 bins for RAD distances (*rd*).
    - HJPD representation (*hj*) trained & tested using 24 bins.
    - HOD representation (*hd*) trained & tested using 14 bins.

    Within each folder is the representation data and grid search outputs (12 files total). The file names for the representations and grid search outputs is shown below:
 
    - rad_d2 --> RAD training representation
    - rad_d2.t --> RAD testing representation
    - rad_d2.out --> RAD grid search output raw data
    - rad_d2.png --> RAD grid search plot
    - hjpd_d2 --> HJPD training representation
    - hjpd_d2.t --> HJPD testing representation
    - hjpd_d2.out --> HJPD grid search output raw data
    - hjpd_d2.png --> HJPD grid search plot
    - hod_d2 --> HOD training representation
    - hod_d2.t --> HOD testing representation
    - hod_d2.out --> HOD grid search output raw data
    - hod_d2.png --> HOD grid search plot


<h2>HOW TO RUN CODE</h2>

I chose to program this project using Python version 3.8. To run the code for this project, follow the steps below:

1. Save the folder named *D2_casey_duncan* to a desired directory in on your computer and use the Ubuntu terminal to navigate into this directory.

2. Install the following required Python packages to run this code.

    - pathlib
    - math
    - copy
    - numpy
    - pandas
    - os
    - sys
    - argparse

    In addition to the packages above, you need to install the libsvm python tools (Download at https://www.csie.ntu.edu.tw/~cjlin/libsvm/). If you download these from the provided link yourself, ensure to follow the following steps to ensure they run with my provided code:
    1. Download the libsvm package as a .zip or .tar.gz file from the link above and save into the *D2_casey_duncan* directory.
    2. Unpack the files so the libsvm package is contained within a new folder within the *D2_casey_duncan* directory. When I downloaded this package, the latest released version for libsvm was version 3.24 so the folder name containing all the libsvm files was called *libsvm-3.24*.
    3. Within the terminal, navigate into the *libsvm-3.24* directory and enter `make` into the terminal so that the libsvm python files can be used.

    I use poetry on my computer to setup python virtual environments for different projects. If you have poetry installed on your computer, you can install the above packages (not including libsvm) by entering `poetry install` into the terminal since I included the .lock and .toml files within the *D2_casey_duncan* folder. If you do not have poetry, install the packages above into your python enviroment using your prefered method.

3. Now run the code using the Command Line Interface (CLI) I have created. The code is currently setup to allow the user to change the bin size, C value, and gamma value for each representation from the terminal. To see the optional arguments, run the code by doing one of the following:
    - If using the poetry environment, enter the `poetry run python main.py -h` into the terminal.
    - If using another method for running python code, run `python main.py -h` in the terminal.

    If done successfully, the following optional arguments should be shown in the terminal:
    ```
	usage: main.py [-h] [-n] [-t] [--rad-ang-bins] [--rad-dist-bins] [--hjpd-bins]
		[--hod-bins] [--rad-c] [--rad-gamma] [--hjpd-c] [--hjpd-gamma] [--hod-c]
		[--hod-gamma]

	optional arguments:
	-h, --help       show this help message and exit
	-n, --new        Generates new representations from raw data.
	-t, --train      Trains RAD, HJPD, & HOD models using RAD, HJPD, & HOD representations.
	--rad-ang-bins   Manually set the bin number for the RAD angle histogram. Default bin
		    size is 11.
	--rad-dist-bins  Manually set the bin number for the RAD distance histogram. Default bin
		    size is 11.
	--hjpd-bins      Manually set the bin number for the HJPD histogram. Default bin size is
		    11.
	--hod-bins       Manually set the bin number for the HOD histogram. Default bin size is
		    22.
	--rad-c          Sets the C value for the RAD SVM training. Default C value is 8.0.
	--rad-gamma      Sets the gamma value for the RAD SVM training. Default gamma value is
		    0.03125.
	--hjpd-c         Sets the C value for the HJPD SVM training. Default C valueis 2.0.
	--hjpd-gamma     Sets the gamma value for the HJPD SVM training. Default gamma value is
		    0.125.
	--hod-c          Sets the C value for the HOD SVM training. Default C value is 2.0.
	--hod-gamma      Sets the gamma value for the HOD SVM training. Default gamma value
    ```

    1. Generating Representations & Perform Grid Search: <br />
    If you would like to generate each representation for the training & testing data using the default bin size, run `main.py -n` using your desired method for running python code. If you would like to change the bin size to something different from the default for a specific representation, type in the argument after the `-n` for the histogram bin size you would like to change. For example, if you would like to change the RAD distance histogram bin size, type in `main.py -n --rad-dist-bins` and follow the instructions that appear in the terminal to change the bin size. Make sure to separate each additonal argument passed with a space. If you just chose to pass the `-n` argument, the representations will begin to genererate. If you chose the change the bin sizes, the representations will begin to generate once the bin sizes are selected. After the representations are generated, the program will automatically perform a grid search on the generated representations. Once the program has finished, verify all representation files have been saved. Once completed, the terminal will display: <br />
    `All Representation files & Grid Search files saved! All saved files can be found at:
    *directory to saved files*`. <br />
    Verify that all files are present in the *directory to saved files*.

    2. Generating Representations, Perform Grid Search, & Training Model: <br />
    If you would like to train the model in addition to generating the representations and performing a grid search, run `main.py -n -t` using your desired method for running python code. Running this will use default values for all representations and default C values & gamma values for all trainings. To change the bin sizes, follow the method explained in step (a) above. To change the C & gamma values, type in the argument after the `-n -t` for the hyperparameter value you would like to change. For example, if you would like to change the RAD gamma value, type in `main.py -n --rad-gamma` and follow the instructions that appear in the terminal to change the gamma value. Make sure to separate each additonal argument passed with a space. Once the program has finished, the model accuracy (using the test data) and confusion matrix for each of the representations will appear in the terminal.


<h2>SOLUTION DESCRIPTION</h2>
The pseudo code for generating each of the three representations is shown below:

<h3>Relative Angles and Distances (RAD)</h3>

1. For each data file, compute the following:
	1. For each video frame, compute the following:
		1. Import Data and save (x,y,z) joint poisitons for the following joints:
			- Joint 1: Hip Center
			- Joint 4: Head
			- Joint 8: Hand Right
			- Joint 12: Hand Left
			- Joint 16: Foot Right
			- Joint 20: Foot Left
		2. Calculate distance vector between joint *i* and reference joint *r*. I selected to use the hip center (joint 1) as the reference joint. The equation for calculating the distance vector is shown below: <br />
		`distance_vector = (x_i - x_r, y_i - y_r, z_i - z_r)`
		3. Calculate angles between the following distance vectors:
			- angle between center-to-head vector & center-to-right-hand vector
			- angle between center-to-head vector & center-to-left-hand vector
			- angle between center-to-right-hand vector & center-to-right-foot vector
			- angle between center-to-left-hand vector & center-to-left-foot vector
			- angle between center-to-right-foot vector & center-to-left-foot vector
		4. Calculate euclidean distance between selected reference joint and each other joint *i*. These distances are defined below:
			- distance from center to head
			- distance from center to right hand
			- distance from center to left hand
			- distance from center to right foot
			- distance from center to left hand
	2. Find the minimum and maximum values for each angle type and distance type in all frames.
2. Find the minimum and maximum values for each angle type and distance type in all data files.
3. For each data file, compute the following:
	1. Save each angle type for all frames to its own histogram with the range being from the minimum value of that angle type to the maximum value of that angle type for all frames and using the user selected bin size. I selected 10 bins for this exercise. There will be 5 different angle histograms, one for each angle type.
	2. Save each distance type for all frames to its own histogram with the range being from the minimum value of that distance type to the maximum value of that distance type for all frames and using the user selected bin size. I selected 10 bins for this exercise. There will be 5 different distance histograms, one for each distance type.
	3. Normalize the data for each of the angle and distance histograms based on the number of frames in each data file.
	4. Combine all 10 histograms into one by concatenating the histogram all together, one after the other.
4. Save the 10 combined histograms for each file as *rad_d1* for the training data and *rad_d1.t* for the testing data. Data is saved in svm format.

<h3>Histogram of Joint Position Differences (HJPD)</h3>

1. For each data file, compute the following:
	1. For each video frame, compute the following:
		1. Import Data and save (x,y,z) joint poisitons for all 20 joints.
		2. Calculate distance vector between joint *i* and reference joint *r*. I selected to use the hip center (joint 1) as the reference joint. The equation for calculating the distance vector is shown below: <br />
		`distance_vector = (x_i - x_r, y_i - y_r, z_i - z_r)`
		3. Save each distance measurement into a vector. There will be 57 measurements in total, since there are 20 joints and there are 3 measurements (delta_x, delta_y, delta_z) per joint.
	2. Find the minimum and maximum values for each distance type in all frames.
2. Find the minimum and maximum values for each angle type and distance type in all data files.
3. For each data file, compute the following:
	1. Save each distance type for all frames to its own histogram with the range being from the minimum value of that distance type to the maximum value of that distance type for all frames and using the user selected bin size. I selected 10 bins for this exercise. There will be 57 different distance histograms, one for each distance type.
	2. Normalize the data for each of the distance histograms based on the number of frames in each data file.
	3. Combine the 3 (delta_x, detla_y, delta_z) histograms for comparing one joint pair into one histogram by concatenating the histogram all together, one after the other. In total, there should be 19 histograms now, one for each joint pair.
4. Save each of the 19 separated histograms for each file as *hjpd_d1* for the training data and *hjpd_d1.t* for the testing data. Data is saved in svm format.

<h3>Histogram of Oriented Displacements (HOD)</h3>

1. For each data file, compute the following:
	1. For each video frame, compute the following:
		1. Import Data and save (x,y,z) joint poisitons for all 20 joints.
		2. Calculate distance vector between each joint in frame *n* and the same joint in frame *n+1*. The equation for calculating the distance vector is shown below: <br />
        	`distance_vector = (x_n - x_n+1, y_n - y_n+1, z_n - z_n+1)`
		3. Calculate the angle in the principle plane projections for each joint from frame *n* to frame *n+1*. The equations for this is shown below: <br />
        	`angle_xy = atan2(distance_vector[1], distance_vector[0])`
        	`angle_yz = atan2(distance_vector[2], distance_vector[1])`
        	`angle_xz = atan2(distance_vector[2], distance_vector[0])`
		4. Save angle in the principle plane projections from frame *n* to frame *n+1* for all joints and all frames. Because there are 20 joints and each joint has 3 angles in the principle plane projections, there should be 60 different angle measurements per frame (n, n+1) pair.
	2. Create the following histograms for each of the 60 angle types:
		1. Histogram capturing data in all frames
		2. Histogram capturing data in first half of all frames
		3. Histogram capturing data in second half of all frames
		4. Histogram capturing data in first quarter of all frames
		5. Histogram capturing data in second quarter of all frames
		6. Histogram capturing data in third quarter of all frames
		7. Histogram capturing data in fourth quarter of all frames
	    The range for each histogram should be between -pi radians and +pi radians since the angles can vary between these two values. The bin size for each histogram is chosen by the user. I selected 10 bins for this exercise.
	3. Normalize the data for each of the histograms based on the number of frames that make up each histogram.
	4. Cocatenate all 7 histograms together into a single vector of histogram values.
2. Save the 7 combined histograms for each file as *hod_d1* for the training data and *hod_d1.t* for the testing data. Data is saved in svm format.
