"""
mic_read.py
Microphone controller module for the Live Spectrogram project, a real time spectrogram visualization tool
"""
############### Import Libraries ###############
import pyaudio
import numpy as np
import matplotlib.pyplot as plt
import wave
import os.path
import warnings
warnings.simplefilter("ignore", UserWarning)
############### Import Modules ###############
from src import file_read_write
from src import constant as c

############### Variables ###############
temp_frame = []      # temporarily store audio for recording wav file
pa = pyaudio.PyAudio()

############### Functions ###############
"""
open_mic:
    creates a PyAudio object and initializes the mic stream
    inputs: none
    ouputs: stream, PyAudio object
"""
def open_mic():
    stream = pa.open(format = c.FORMAT, channels = c.MIC_CHANNELS, rate = c.SAMPLE_RATE,
                     input = True, frames_per_buffer = c.CHUNK_SIZE)
    return stream,pa

"""
get_data:
    reads from the audio stream for a constant length of time, converts it to data
    inputs: stream, PyAudio object
    outputs: int16 data array
"""
def get_data(stream,pa):
    frame = stream.read(c.CHUNK_SIZE)
    data = np.fromstring(frame,np.int16)
    return data, frame

"""
store_temp_frame:
    accumulate data frame temporarily for later writing in wav
"""
def store_temp_frame(frame):
    temp_frame.append(frame)

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
write_wav:
    create a .wav file to record data
    inputs: label
"""
def write_wav(label):
    file_name = c.WAV_FILE_PATH + get_file_name(label) + '.wav'
    waveFile = wave.open(file_name, 'wb')
    waveFile.setnchannels(c.MIC_CHANNELS)
    waveFile.setsampwidth(pa.get_sample_size(c.FORMAT))
    waveFile.setframerate(c.SAMPLE_RATE)
    waveFile.writeframes(b''.join(temp_frame))
    print(file_name, "saved")
    waveFile.close()
    temp_frame.clear()

############### Test Functions ###############
"""
make_10k:
    creates a 10kHz test tone
"""
def make_10k():
    x = np.linspace(-2*np.pi,2*np.pi,21000)
    x = np.tile(x,int(SAMPLE_LENGTH/(4*np.pi)))
    y = np.sin(2*np.pi*5000*x)
    return x,y

"""
show_freq:
    plots the test tone for a sanity check
"""
def show_freq():
    x,y = make_10k()
    plt.plot(x,y)
    plt.show()
    
