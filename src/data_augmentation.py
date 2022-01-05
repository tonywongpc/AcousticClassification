from pydub import AudioSegment
import os
from scipy import rand
import scipy.signal
import scipy.io.wavfile

DATASET_PATH = "dataset/dataset_Bag_Air/data/"
OUTPUT_PATH = "dataset/dataset_Bag_Air/data_augmented/"
NOISE_CITY_FILE = "dataset/noise_wav/city_noise_2min.wav"
NOISE_OFFICE_FILE = "dataset/noise_wav/office_noise_2min.wav"
LENGTH_OF_RECORDING = 1400  # 1.4s wav recording
TEXTURE_NAME = ['cardboard', 'cotton', 'foam', 'linen', 'nylon', 'paper', 'polyester', 'tin']
GESTURE_NAME = ['circle', 'cross', 'rectangle', 'rub', 'star', 'swipe',  'tick', 'triangle']
TIME_SHIFT_TIME = 10    # number of time-shift perform per sound file (10 mean output 10 augmented files per input sound)

# set ffmpeg for AudioSegment
AudioSegment.ffmpeg = os.getcwd()+"\\FFmpeg\\bin\\ffmpeg.exe"

def augment_sound(in_file):
    sound = AudioSegment.from_file(DATASET_PATH + in_file)

    for i in range(TIME_SHIFT_TIME):
        # find peaks
        s = scipy.io.wavfile.read(DATASET_PATH + in_file)
        indexes, _ = scipy.signal.find_peaks(s[1], height=10000, distance=2.1)  # get a list of peaks
        first_peak_time = indexes[0]/48000*1000
        last_peak_time = indexes[-1]/48000*1000
        peaks_center = (first_peak_time + last_peak_time)/2

        # range 1.4 (0 to 1400)
        sound_length = 1000
        pos_start = int(rand()*(LENGTH_OF_RECORDING - sound_length))
        pos_end = pos_start + sound_length
        s1 = sound[pos_start:pos_end]

        # randomily determine whether adding noise or not  (50%)
        is_add_noise = False
        if(int(rand() * 2)==0):
            is_add_noise = True
            noise_reduce_factor = rand() * 6 # reduce noise by 0 - 6dB
            # randomily determine to add city or office noise
            noise = noise_city
            if (int(rand() * 2) == 0):
                noise = noise_office
            # random noise position
            noise_start = int(rand()*(120000 - sound_length))
            noise_end = noise_start + sound_length
            # mix sound with noise
            s2 = noise[noise_start:noise_end] - noise_reduce_factor
            s1 = s1.overlay(noise)

        file_name = OUTPUT_PATH + in_file[:-4] + "_aug_" + str(i+1) + ".wav"
        s1.export(file_name, format='wav')


noise_office = AudioSegment.from_file(NOISE_OFFICE_FILE)
noise_city = AudioSegment.from_file(NOISE_OFFICE_FILE)
augment_sound("cardboard/circle/data00082_cardboard_circle.wav")
