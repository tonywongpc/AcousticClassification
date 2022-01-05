############### Import Libraries ###############
import os.path
import math
import warnings
warnings.simplefilter("ignore", UserWarning)

############### Import Modules ###############
from src import keyboard_read
from src import constant as c

################ Variables ###############
global pred_data         #store the data temperarily for prediction
global pred_frame       # store the (binary) data temparily for prediction
global recorded_chunk_num  #number of already recorded for prediction
global pred_chunk_num  #number of already recorded for prediction

# initialize the variables
pred_data = []
pred_data2 = []
recorded_chunk_num = 0
pred_chunk_num = 0

############### Functions ###############
"""
get_file_name: 
    return the new file name for next recording (without extension e.g. ".wav")
    inputs:  label of the gesture
    outputs: file name
"""
def get_file_name(label):
    fileCounter =  len(os.listdir(c.WAV_FILE_PATH)) + 1
    # add '0' before the number for formatting (e.g. data00001)
    number = str(fileCounter)
    while len(number) < 5:
        number = '0' + number
    # finalize the file name
    name = 'data' + number + '_' + label
    return name

"""
store_pred_data:
    Accumulate data temporarily for later recording as wav or prediction.
    For prediction, pred_data2 store data with delay of half of the prediction data length.
    e.g. the system predict a gesture with 1s sound data, pred_data and pred_data2 store sound data with a 0.5s difference 
"""
def store_pred_data(data):
    # store data in pred_data
    for d in data:
        pred_data.append(d)

    # store data in pred_data2
    if c.IS_AUTO_PREDICT == True:
        if len(pred_data) > (c.SAMPLE_RATE * c.WAIT_TIME)/2 or len(pred_data2)>0:
            for d in data:
                pred_data2.append(d)

############# Helper Functions #############
"""
str_list_to_int_list: 
    a helper function converting a string of the entire list 
    input:  a string of list (i.e. "0,1,0,3,2") 
    output: a list of integer (i.e. [0,1,0,3,2])
"""
def str_list_to_int_list(list_string):
    values = []
    values_str = list_string[0].split(',')
    for j in range(len(values_str)):
        if values_str[j] != '':
            values.append(int(values_str[j]))  # append an integer from time series data
    return values