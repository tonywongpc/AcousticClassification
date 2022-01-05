# AcousticClassification

This repository contains dataset and code for classifying input on textiles.

Dataset contains 60 samples per class and each sample is a 1.4s recording of the sound generated when using the switch. I extracted MFCC (Mel-frequency cepstral coefficients) images for each sample. I then used a pre-trained model of VGG-19 and extracted abstract features of the MFCC image from the flatten layer. After extracting these features, I created a 70–30 train test split and trained a LinearSVM for classification.

The 6 classes of mechanical switches: pushButton, rest, rockerSwitch, rollerLever, rotarySwitch, screwCap, toggleSwitch

Functions
The main.py contains three functions

extract_plots_from_audio extracts waveform and MFCC plots from audio files in the dataset and store the plots.
dataset name name of dataset.
train_model extracts abstract features of MFCC image with pre-trained VGG-19 model and then train a LinearSVM model for classification.
dataset name name of dataset.
plot_name use waveform (plot) / MFCC images (plot_mfcc) for training.
save_model_path path to save trained model.
test_model loads the trained model and predicts result with a test sample file.
model_path path of trained model.
test_sample_path path of sample file to test.
How to use
To train a classification model, run extract_plots_from_audio and then train_model. To test a trained model, run test_model.

Downlaod dataset
https://drive.google.com/file/d/1-uEwkkqqUfV-UJWoPZLBkGypBQHbEkt-/view?usp=sharing

Please put the download dataset inside the dataset folder.
