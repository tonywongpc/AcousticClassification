"""
This is a python containing all constant for
"""
import pyaudio
import math
from scipy import signal

#--------------------- Paths and Class --------------------------
MODE = 'TGR' # Texture, Bag, TGR

WAV_FILE_PATH = '../recorded_data/'         # path for recording wav files
#DATA_PATH = '../dataset/TGR_texture/'   # path of dataset (wav files) for model training
MODEL_PATH = "../model/mfcc_TGR_texture.sav"        # path to save trained model
CLASS_NAME = ["cotton",  "foam", "nylon", "linen", "polyester", "cardboard"]   #TGR texture

if MODE == 'Texture':
    MODEL_PATH = "../model/mfcc_TGR_texture.sav"
    CLASS_NAME = ["cotton", "foam", "nylon", "linen", "polyester", "cardboard", "rest"]
elif MODE == 'Bag':
    MODEL_PATH = "../model/mfcc_bag.sav"
    CLASS_NAME = ["lid",  "zip", "zip", "tap", "rest"]
elif MODE == 'TGR':
    MODEL_PATH = "../model/mfcc_TGR_texture.sav"
    CLASS_NAME = ["cotton", "foam", "nylon", "linen", "polyester", "cardboard"]
    # list of trained model paths (gesture classifier for each texture), used for texture-gesture recognition
    MODEL_PATH2 = ["../model/mfcc_cotton_gesture.sav",
                   "../model/mfcc_foam_gesture.sav",
                   "../model/mfcc_nylon_gesture.sav",
                   "../model/mfcc_linen_gesture.sav",
                   "../model/mfcc_polyester_gesture.sav",
                   "../model/mfcc_cardboard_gesture.sav"]
    GESTURE_NAME = ["Swipe", "Circle", "Triangle", "Rub"]

#------------------- Prediction -------------------------
IS_TEST_MSG = False              # True for testing result message
IS_PREDICT_ON = True             # True for real-time recognition
IS_AUTO_PREDICT = False          # True for auto prediction, IS_PREDICT_ON must be True
#IS_FILTER_ON = True             # Ture for turning on filter
#IS_SEGMENTATION_ON = True       # True for turning on segmentation
#SEGMENTED_SIGNAL_LENGTH = 1     # length of segmented signal: 1s
CUTOFF_FREQ = 3000               # cutoff frequency for high-pass filter
TEST_MSG = ['clap', 'tap', 'swipe', ' ']    # result messages showing after prediction (for testing)
MSG_FONT_SIZE = 50              # font size of result message

#-------------------- Mircophone ------------------------
SAMPLE_RATE = 48000         #sample rate
WAIT_TIME = 1.4             # recorded signal length in second
FORMAT = pyaudio.paInt16    #conversion format for PyAudio stream
MIC_CHANNELS = 1            #microphone audio channels
CHUNK_SIZE = 4000           #number of samples to take per read
SAMPLES_PER_FRAME = 12      # Number of mic reads concatenated within a single window1111111
NFFT = 512                  # NFFT value for spectrogram, 256, 512, 1024
OVERLAP = 250               # overlap value for spectrogram, 512, 1000
SAMPLE_LENGTH = int(CHUNK_SIZE*1000/SAMPLE_RATE)    #length of each sample in ms
MAX_CHUNK = math.ceil((SAMPLE_RATE*WAIT_TIME)/CHUNK_SIZE)     # maximum of chunk containing complete sound data
DELAY_TIME_BEFORE_RECORDING = 0.5       # wait time (in second) before starting to record
#filter_coef = signal.firwin(101, cutoff=CUTOFF_FREQ, fs=SAMPLE_RATE, pass_zero=False) #coefficient vector for signal filtering

# -------------- Keyboard and Recording ------------------
LABEL_INDICES = [1, 2, 3, 4, 5, 6, 0]   # label indices used for training and classification, we use '0' for rest (background noise)
INPUT_KEYS = ['1', '2', '3', '4', '5', '6', '`']   #keyboard key trigger recording
#LABELS = ["cotton",  "foam", "nylon", "linen", "polyester", "cardboard", "rest"] #name appends to recording wav files (each label represent one class)
LABELS = ["lid",  "zip_open", "zip_close", "tap", "rest"]
