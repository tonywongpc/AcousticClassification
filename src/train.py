"""
Train a SVM classifier with VGG19 model as feature extractor
"""
# import library
from scipy.io.wavfile import read
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
from matplotlib import cm
from os import walk
import itertools
import os
import numpy as np
import tensorflow.keras
from python_speech_features import mfcc
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.models import Model
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
import pickle
from datetime import datetime
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation

# import modules
import src.constant as c

def time_shift(data, sampling_rate, shift_max):
    shift = np.random.randint(sampling_rate * shift_max)
    direction = np.random.randint(0, 2)
    if direction == 1:
            shift = -shift
    augmented_data = np.roll(data, shift)
    # Set to silence for heading/ tailing
    if shift > 0:
        augmented_data[:shift] = 0
    else:
        augmented_data[shift:] = 0
    return augmented_data

def extract_plots_from_audio():
    # create folder if not exist
    if not os.path.exists("../dataset/" + dataset + "/plot/"):
        os.makedirs("../dataset/" + dataset + "/plot/")
    if not os.path.exists("../dataset/" + dataset + "/plot_mfcc/"):
        os.makedirs("../dataset/" + dataset + "/plot_mfcc/")

    for c in CLASS_NAME:
        class_name = c
        # get wave file names
        wavs = []
        for (_,_,filenames) in walk("../dataset/" + dataset + "/data/" + class_name + '/'):
            wavs.extend(filenames)
            break
        print(c, len(wavs))

        if not os.path.exists("../dataset/" + dataset + "/plot/" + class_name + '/'):
            os.makedirs("../dataset/" + dataset + "/plot/" + class_name + '/')
        if not os.path.exists("../dataset/" + dataset + "/plot_mfcc/" + class_name + '/'):
            os.makedirs("../dataset/" + dataset + "/plot_mfcc/" + class_name + '/')

        for w in wavs:
            # -------------- plot waveform --------------
            input_data = read("../dataset/" + dataset + "/data/" + class_name + '/' + w)
            audio = input_data[1]
            plt.plot(audio)
            plt.ylabel("Amplitude")
            plt.xlabel("Time")
            #plt.show()
            plt.savefig("../dataset/" + dataset + "/plot/" + class_name + '/' + w.split('.')[0] + '.png')

            # -------------- plot mfcc  -----------
            (rate,sig) = wav.read("../dataset/" + dataset + "/data/" + class_name + '/' + w)
            mfcc_feat = mfcc(sig,rate)
            fig, ax = plt.subplots()
            mfcc_data= np.swapaxes(mfcc_feat, 0 ,1)
            cax = ax.imshow(mfcc_data, interpolation='nearest', cmap=cm.coolwarm, origin='lower', aspect='auto')
            #plt.show()
            plt.savefig("../dataset/" + dataset + "/plot_mfcc/" + class_name + '/' + w.split('.')[0] + '.png')
            plt.close('all')

            # -------------- plot waveform time shifted --------------
            audio_time_shifted = time_shift(input_data[1], 48000, 0.4)

            plt.plot(audio_time_shifted)
            plt.ylabel("Amplitude")
            plt.xlabel("Time")
            # plt.show()
            plt.savefig("../dataset/" + dataset + "/plot/" + class_name + '/' + w.split('.')[0] + '-a.png')

            # -------------- plot mfcc time shifted -----------
            (rate, sig) = wav.read("../dataset/" + dataset + "/data/" + class_name + '/' + w)
            mfcc_feat = mfcc(audio_time_shifted, rate)
            fig, ax = plt.subplots()
            mfcc_data = np.swapaxes(mfcc_feat, 0, 1)
            cax = ax.imshow(mfcc_data, interpolation='nearest', cmap=cm.coolwarm, origin='lower', aspect='auto')
            # plt.show()
            plt.savefig("../dataset/" + dataset + "/plot_mfcc/" + class_name + '/' + w.split('.')[0] + '-a.png')
            plt.close('all')


def get_features(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    flatten = model.predict(x)
    return list(flatten[0])


#plot_confusion_matrix:
#    - This function prints and plots the confusion matrix.
#    - Normalization can be applied by setting `normalize=True`.
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix',cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, "{:.2f}".format(cm[i, j]),
                 horizontalalignment="center",
                 fontsize=10,
                 color="white" if cm[i, j] > thresh else "black")   # color of font

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.xticks(rotation=45)


def train_model(plot_name = "plot_mfcc", save_model_path=""):
    X_train, X_test, y_train, y_test = get_train_test_set(plot_name)
    # train a SVC model
    clf = LinearSVC(random_state=0, tol=1e-5)   # define a linear SVC model
    clf.fit(X_train, y_train)
    predicted = clf.predict(X_test)
    print(accuracy_score(y_test, predicted))  # print the accuracy

    # save the trained model
    # refer to https://machinelearningmastery.com/save-load-machine-learning-models-python-scikit-learn/
    if len(save_model_path) > 0:
        pickle.dump(clf, open(save_model_path, 'wb'))

    # Plot normalized confusion matrix
    cm = confusion_matrix(y_test, predicted)  # get confusion matrix
    plt.figure()
    plot_confusion_matrix(cm, classes=CLASS_NAME, normalize=True)
    plt.show()

def train_model_keras(plot_name="plot_mfcc", save_model_path=""):
    model = Sequential()
    model.add(Dense(64, activation='relu'))
    model.add(Dense(7, kernel_regularizer=l2(0.01)))
    model.add(Activation('softmax'))
    model.compile(loss='squared_hinge',
                  optimizer='adadelta',
                  metrics=['accuracy'])
    X_train, X_test, y_train, y_test = get_train_test_set(plot_name)
    model.fit(X_train, y_train)
    predicted = model.predict(X_test)
    print(accuracy_score(y_test, predicted))  # print the accuracy
    model.save(save_model_path)


def get_train_test_set(plot_name = "plot_mfcc"):
    # define feature extractor
    base_model = VGG19(weights='imagenet')
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('flatten').output)

    # get features from plots
    X = []
    y = []
    for c in CLASS_NAME:
        plots = []
        for (_, _, filenames) in os.walk("../dataset/" + dataset + "/" + plot_name + "/" + c + "/"):
            plots.extend(filenames)
            break
        print(len(plots), plots)
        for cplot in plots:
            X.append(get_features("../dataset/" + dataset + "/" + plot_name + "/" + c + "/" + cplot, model))
            y.append(CLASS_NAME.index(c))
    # train the model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
    return X_train, X_test, y_train, y_test


def test_model(plot_name="plot_mfcc", model_path=""):
    if len(model_path) > 0:
        print
        X_train, X_test, y_train, y_test = get_train_test_set(plot_name)
        trained_model = pickle.load(open(model_path, 'rb'))  # load trained model
        predicted = trained_model.predict(X_test)
        print(accuracy_score(y_test, predicted))  # print the accuracy
        cm = confusion_matrix(y_test, predicted)  # get confusion matrix
        plt.figure()
        plot_confusion_matrix(cm, classes=CLASS_NAME, normalize=True)
        plt.show()

# this is a function for extraction waveform plot and MFCC plot for texture-gesture dataset
def extract_plots_from_audio_textureGesture():
    TEXTURE_NAME = ['cardboard', 'cotton', 'foam', 'linen', 'nylon', 'paper', 'polyester', 'tin']
    GESTURE_NAME = ['circle', 'cross', 'rectangle', 'rub', 'star', 'swipe',  'tick', 'triangle']

    for t in TEXTURE_NAME:
        CLASS_NAME = []
        for g in GESTURE_NAME:
            CLASS_NAME.append(t + '/' + g)
        print(CLASS_NAME)
        extract_plots_from_audio()


# -----------------------------------------------------------------
CLASS_NAME = c.CLASS_NAME
dataset = "TGR_gestures_polyester"

#extract_plots_from_audio()
train_model(plot_name="plot_mfcc", save_model_path="../model/mfcc_polyester_gesture.sav")
#test_model(plot_name="plot_mfcc", model_path = "../model/mfcc_6texture_augmented2.sav")
