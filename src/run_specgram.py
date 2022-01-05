"""
run_specgram.py
Modified By Tony Wong (tonywongpc@gmail.com)
from the code created By Alexander Yared (akyared@gmail.com)
Main Script for the Live Spectrogram project, a real time spectrogram visualization tool
"""
############### Import Libraries ###############
from matplotlib.image import AxesImage
from matplotlib.mlab import window_hanning,specgram
from matplotlib.colors import LogNorm
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import warnings
warnings.simplefilter("ignore", UserWarning)

############### Import Modules ###############
from src import mic_read
from src import keyboard_read
from src import file_read_write
from src import classifier
from src import constant as c

############### Functions ###############
"""
get_sample:
    - gets the audio data from the microphone and handle the data recording and gesture prediction
    - inputs: audio stream and PyAudio object
    - outputs: int16 array
"""
def get_sample(stream,pa):
    data, frame = mic_read.get_data(stream,pa)
    ############### Data recording when keypressed ###############
    if keyboard.liveRecordOn:
        mic_read.store_temp_frame(frame)        # record data for writing wav file
        file_read_write.recorded_chunk_num += 1
        ######### record enough data and write to csv file ########
        if file_read_write.recorded_chunk_num >= c.MAX_CHUNK:
            if keyboard.liveRecordOn:
                mic_read.write_wav(keyboard.label)
                keyboard.liveRecordOn = False
            keyboard.writeLock = False  # unlock the keyboard for next input

    ################ Record data and prediction #############
    if c.IS_PREDICT_ON:
        # handle prediction data when auto prediction is OFF
        if c.IS_AUTO_PREDICT== False:
            if keyboard.predictionOn == True:   # check if Space is pressed
                if file_read_write.pred_chunk_num < c.MAX_CHUNK + 1:
                    file_read_write.store_pred_data(data)
                    file_read_write.pred_chunk_num += 1
                else:
                    clf.predict_mfcc()   # start prediction once enough data is stored
                    keyboard.predictionOn = False   # reset the state for next prediction
        # handle prediction data when auto prediction is ON
        else:
            if file_read_write.pred_chunk_num < c.MAX_CHUNK + 1:
                """do in a thread"""  # repeatedly predict once enough data is captured
                # record data for prediction
                file_read_write.store_pred_data(data)
                file_read_write.pred_chunk_num += 1
    return data

"""
get_specgram:
    - takes the FFT to create a spectrogram of the given audio signal
    - input: audio signal, sampling rate
    - output: 2D Spectrogram Array, Frequency Array, Bin Array
"""
def get_specgram(signal,rate):
    arr2D,freqs,bins = specgram(signal, window=window_hanning, Fs=rate,
                                NFFT=c.NFFT, noverlap=c.OVERLAP)
    return arr2D,freqs,bins

"""
update_fig:
    - updates the image, just adds on samples at the start until the maximum size is
      reached, at which point it 'scrolls' horizontally by determining how much of the
      data needs to stay, shifting it left, and appending the new data. 
    - inputs: iteration number
    - outputs: updated image
"""
def update_fig(n):
    data = get_sample(stream,pa)
    arr2D,freqs,bins = get_specgram(data, c.SAMPLE_RATE)
    im_data = im.get_array()
    # update live spectrogram of input sound data
    if n < c.SAMPLES_PER_FRAME:
        im_data = np.hstack((im_data,arr2D))
        im.set_array(im_data)
    else:
        keep_block = arr2D.shape[1]*(c.SAMPLES_PER_FRAME - 1)
        im_data = np.delete(im_data,np.s_[:-keep_block],1)  # remove old data
        im_data = np.hstack((im_data,arr2D))        # append new data
        im.set_array(im_data)
        # show prediction result
        if c.IS_PREDICT_ON:
            plt.figure(1)
            # clear old text message
            if clf.result_txt != None:
                clf.result_txt.remove()
            # write new text message
            clf.result_txt = plt.text(0.4, 0.5, clf.predict_message, fontsize=c.MSG_FONT_SIZE,
                                      horizontalalignment='center', verticalalignment = 'center')
    # update subplot 2
    #im2.set_array(im_data)     # use data from subplot 1
    return im,

"""
main:
    - plot live spectrogram from microphone data 
"""
def main():
    ############### Initialize Keyboard ###########
    global keyboard
    keyboard = keyboard_read.keyboard_read()
    keyboard.keybaord_start_thread()

    ############## Initialize classifier ##########
    if c.IS_PREDICT_ON:
        global clf
        clf = classifier.classifier()
        if c.IS_AUTO_PREDICT == True:
            clf.predict_start_thread()

    ############### Initialize Plot ###############
    global stream, pa, im, im2
    fig = plt.figure(1)
    # Launch the stream and the original spectrogram
    stream,pa = mic_read.open_mic()
    data = get_sample(stream,pa)
    arr2D,freqs,bins = get_specgram(data, c.SAMPLE_RATE)
    extent = (bins[0],bins[-1]*c.SAMPLES_PER_FRAME,freqs[-1],freqs[0])

    # initialize subplot 1
    plt.subplot(211)
    im = plt.imshow(arr2D,aspect='auto',extent = extent,interpolation="none",
                    cmap = 'viridis',norm = LogNorm(vmin=.01,vmax=1))
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.title('Real Time Spectogram')
    plt.gca().invert_yaxis()

    # initialize subplot 2
    plt.subplot(212)
    """
    im2 = plt.imshow(arr2D,aspect='auto',extent = extent,interpolation="none",
                     cmap = 'viridis',norm = LogNorm(vmin=.01,vmax=1))
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    """
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.axis('off')
    #plt.colorbar() #enable if you want to display a color bar

    ############### Animate ###############
    anim = animation.FuncAnimation(fig,update_fig, blit = False, interval=c.CHUNK_SIZE/1000)
    try:        plt.show()
    except:     print("Plot Closed")

    ############### Terminate ###############
    stream.stop_stream()
    stream.close()
    pa.terminate()
    print("Program Terminated")

################### Main Program #####################
if __name__ == "__main__":
    main()