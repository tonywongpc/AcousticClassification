############### Import Libraries ###############
import tensorflow as tf
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
from matplotlib import cm
from python_speech_features import mfcc
from datetime import datetime
from tensorflow.keras.models import Model, load_model
from kapre.time_frequency import Melspectrogram, Spectrogram
from threading import Thread, Timer
import os
import warnings
warnings.simplefilter("ignore", UserWarning)
import pickle
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import preprocess_input

############### Import Modules ###############
from src import file_read_write
from src import helper_functions
from src import constant as c

##################### Class #####################
class classifier:
    """
    constructor:
        - initialize variables for real-time classification and load the trained model
    """
    def __init__(self):
        self.thread = None
        self.temp_data = []    # store temp data for showing spectrogram of input signal
        self.is_temp_data_updated = False   # boolean to initiate a spectrogram update of new input signal
        self.predict_message = ''

        # variable for VGG19 classification MFCC
        base_model = VGG19(weights='imagenet')
        self.FEmodel = Model(inputs=base_model.input, outputs=base_model.get_layer('flatten').output) # feature extractor
        self.trained_model = pickle.load(open(c.MODEL_PATH, 'rb'))  # load trained model
        self.trained_gesture_models = [] # list of gesture models
        if c.MODE == 'TGR':
            for i in range(len(c.CLASS_NAME)):
                self.trained_gesture_models.append(pickle.load(open(c.MODEL_PATH2[i], 'rb')))
        self.mfcc_img_name = ""
        self.result_txt = None  # plt.text containing result message for drawing on plt screen
        self.txt_count = 0

    """
    predict_start_thread:
        - start a new thread for predicting the gesture for auto prediction (run recursively)
    """
    def predict_start_thread(self):
        Timer(c.WAIT_TIME/2, self.predict_start_thread).start()
        self.predict_mfcc()

    """
    predict_mfcc:
        - perform a classification with load model by VGG19 feature extractor of MFCC image 
        - input:  sensor values in a 3d array format (1, 52920, 3)
        - output: the index of prediction gesture
    """
    def predict_mfcc(self):
        # get audio data
        data_idx = 0
        req_len = int(c.SAMPLE_RATE * c.WAIT_TIME)  # required data length for prediction

        if c.IS_AUTO_PREDICT == False:
            data = file_read_write.pred_data
        else:
            data1 = file_read_write.pred_data
            data2 = file_read_write.pred_data2
            data, data_idx = self.alternateDataStream(data1, data2)  # use data1 and data2 for prediction alternatively

        # break the function if not enough data for prediction
        if (len(data) < req_len):
            # print("not enough data")
            return
        # remove extra input data
        if (len(data) > req_len):
            data = data[:req_len]

        startTime = datetime.now()
        self.stream_data_to_mfcc_image(data)      # convert data to mfcc
        if c.MODE == 'TGR':
            idx, idx2 = self.vgg_feature_extraction_and_prediction_TGR()        # Texture-Gesture classification
        else:
            idx = self.vgg_feature_extraction_and_prediction()        # classification

        # show result message
        if c.MODE == 'TGR':
            pred_label = c.CLASS_NAME[idx] + " " +  c.GESTURE_NAME[idx2]    # get label names for TGR
        else:
            pred_label = c.CLASS_NAME[idx]      # get label name
        print("prediction time: ", datetime.now() - startTime, "\t result: ", pred_label, idx)
        self.predict_message = pred_label   # update result in predict message
        if c.IS_TEST_MSG == True:
            self.predict_message = c.TEST_MSG[self.txt_count]
            self.txt_count = (self.txt_count+1)% len(c.TEST_MSG)    # prevent out of bound

        ########## prepare for next prediction ########
        # reset data for next prediction
        file_read_write.pred_chunk_num = 0
        if data_idx == 2:
            file_read_write.pred_data2.clear()
        else:
            file_read_write.pred_data.clear()
        return

    """
    stream_data_to_mfcc_image
        - This function convert live stream acoustic data to resized MFCC image and save as a png. 
        - input: audio data from data stream
    """
    def stream_data_to_mfcc_image(self, data):
        X = librosa.util.buf_to_float(np.asarray(data))  # convert data to array
        mfcc_feat = mfcc(X, c.SAMPLE_RATE)  # get mfcc feature
        mfcc_data = np.swapaxes(mfcc_feat, 0, 1)
        self.mfcc_img_name = datetime.now().strftime("%Y%m%d-%H%M%S")
        # save MFCC image
        plt.figure('temp')
        fig, ax = plt.subplots()
        cax = ax.imshow(mfcc_data, interpolation='nearest', cmap=cm.coolwarm, origin='lower', aspect='auto')
        # plt.show()     # testing: visual the MFCC image
        fig.savefig('../temp_mfcc_images/' + self.mfcc_img_name + '.png')
        plt.close('temp')

    """
    vgg_feature_extraction_and_prediction
        - This function loads the MFCC image extracted from live stream then performs VGG feature extraction and classification.
        - output: a class index fo predicted result 
    """
    def vgg_feature_extraction_and_prediction(self):
        img_path = '../temp_mfcc_images/'+ self.mfcc_img_name + '.png'
        img = image.load_img(img_path, target_size=(224, 224))  # get MFCC image and resize
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        flatten = self.FEmodel.predict(x)
        vector = list(flatten[0])
        predicted = self.trained_model.predict([vector])  # prediction
        idx = predicted[0]
        return idx

    """
        vgg_feature_extraction_and_prediction
            - This function loads the MFCC image extracted from live stream then performs VGG feature extraction and classification.
            - output: a class index fo predicted result 
        """

    def vgg_feature_extraction_and_prediction_TGR(self):
        img_path = '../temp_mfcc_images/' + self.mfcc_img_name + '.png'
        img = image.load_img(img_path, target_size=(224, 224))  # get MFCC image and resize
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        flatten = self.FEmodel.predict(x)
        vector = list(flatten[0])
        predicted = self.trained_model.predict([vector])  # texture prediction
        idx = predicted[0]
        predicted2 = self.trained_gesture_models[idx].predict([vector])  # gesture prediction
        idx2 = predicted2[0]
        return idx, idx2

    """
    alternateDataStream
        - There are two data stream used for prediction. The two data streams overlap 50% to avoid data missing between
          predictions. This functions determine which data stream (data1 or data2) to be used for each prediction. 
    """
    def alternateDataStream(self, data1, data2):
        # get data index from 2 stream alternatively
        if len(data1) > len(data2):
            data_idx = 1
            data = data1
        else:
            data_idx = 2
            data = data2
        #print('time: ', datetime.now(), '\tdata idx: ', data_idx, '\tdata 1:', len(data1), '\tdata 2:', len(data2))
        return data, data_idx

    """
    clear_temp_data:
        - clear temporary data for plotting real-time spectrogrum
    """
    def clear_temp_data(self):
        self.temp_data = []
