# Park-Ai-Final
manifesting a 100%!!!!!

-Datasets *time intensive*
=> Download following parts from the Luna16 dataset: 1. as many subsets as you want for the scans, 2. seg-lungs-LUNA16 for the masks
- possible sites include https://zenodo.org/records/3723295, https://www.kaggle.com/datasets/fanbyprinciple/luna-lung-cancer-dataset, and the LUNA16 website
=> remove all .mhd files from all folders and combine all subsets into one folder
=> as the provided segmented data does not cover every scan, you will need to use tools like kaleidescope and beyond compare to compare and remove .zraw files from both folders that do not have coresponding scans/masks (probably 500+ files given 2-3 subsets)

- Running
=> separate out a testing set of masks and scans
=> in the code, set up your paths for the scans (subset) and masks (seg-lungs-LUNA16)
=> run the program

Description:
This POC uses a U-net convolutional neural network in a K-fold cross validation approach to design a machine learning model that can accurately identify the lungs on an axial CT scan.