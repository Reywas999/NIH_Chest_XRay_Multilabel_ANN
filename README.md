# NIH_Chest_XRay_Multilabel_ANN
Multilabel ANN created using the NIH chest x-ray images and data as of (08/27/2024).
_________________________________________________________________________________________________________________________________
NOTE: Create a folder in your WD that is named X_Ray_NN. Store the split_train_val folder, data_entry_2017_v2020 csv file, 
and/or trained_model file in this folder.
_________________________________________________________________________________________________________________________________
This personal project was just a proof of concept to get a working model that outputs likelihood values for each of the 14 
potential outcome labels, given an input chest x-ray image (among other inputs). The f-score, precision, and accuracy are 
abysmal. I've also attempted to create an ensemble model by first using a CNN for feature extraction, and then training both a 
KNN and Random Forest classifier on these features, but these both returned only slightly better results.
\
\
WARNING: running "NIH_CXR8_Chest_X_Rays_Functions" will download all 100,000+ x-ray images from the NIH website.
_________________________________________________________________________________________________________________________________
NIH Chest X-ray Dataset of 14 Common Thorax Disease Categories:\
https://nihcc.app.box.com/v/ChestXray-NIHCC/file/371647823217
\1, Atelectasis; \
2, Cardiomegaly; \
3, Effusion; \
4, Infiltration; \
5, Mass; \
6, Nodule; \
7, Pneumonia; \
8, Pneumothorax; \
9, Consolidation; \
10, Edema; \
11, Emphysema; \
12, Fibrosis; \
13, Pleural_Thickening; \
14, Hernia\
\
\
Meta data for all images (Data_Entry_2017_v2020.csv): Image Index, Finding Labels, Follow-up #,
Patient ID, Patient Age, Patient Gender, View Position, Original Image Size and Original Image
Pixel Spacing.
\
\
Two data split files (train_val_list.txt and test_list.txt) are provided. Images in the ChestX-ray
dataset are divided into these two sets on the patient level. All studies from the same patient will
only appear in either training/validation or testing set.
_________________________________________________________________________________________________________________________________
You will need to have the following files in your working directory:
1) requirements.txt --> These are all the required libraries for executing the "NIH_CXR8_Chest_X_Rays_Functions" script.
2) Split_Train_Val --> This folder contains test_list.txt and train_val_list.txt for splitting the data
3) Data_Entry_2017_v2020.csv --> This file contains other patient data (see above)
_________________________________________________________________________________________________________________________________
ALTERNATIVELY:
If you already have a trained model, you can simply run the "NIH_CXR8_Chest_X_Rays_linux.py" script.\
A trained model is also provided.
_________________________________________________________________________________________________________________________________
