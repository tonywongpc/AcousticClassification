import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

import warnings
warnings.simplefilter("ignore", UserWarning)

"""
segmentation:
chop sound signal into a shorter segment starting when first peak appear 
input:
    - X:        sound signal data
    - s_rate:   sampling rate
    - segment_len:  time lengeht of segmented sound signal (in second)
    - wait_time:    total time length of input sound signal 
output:
    - X:        segmented sound signal data
"""
def segmentation(X, s_rate, segment_len = 0.8, wait_time = 1):
    # calculate the position first peak happens
    tempo, beats = librosa.beat.beat_track(y=X, sr=s_rate)
    try:
        first_peak_time = librosa.frames_to_time(beats, sr=s_rate)[0]
    except:
        print("out of bounds error")
        return X
    #print("first_peak time in second: ", first_peak_time)

    # segmentation
    # load the file again and specify the time starting 50ms before first peak to 750ms after first peak
    start_seg_time = first_peak_time - 0.05
    end_seg_time = start_seg_time + segment_len

    # prevent input out of bounce
    if end_seg_time > wait_time or start_seg_time < 0:
        start_seg_time = wait_time - segment_len

    # convert from time to samples
    start_sample = librosa.time_to_samples(start_seg_time, s_rate)
    end_sample = librosa.time_to_samples(end_seg_time, s_rate)

    # get the audio data starting 50ms before first peak to 750ms after first peak
    X = X[start_sample:end_sample]

    # plot segmented signal for testing
    #plt.figure(2)
    #fig, ax = plt.subplots(nrows=3, sharex=True, sharey=True)
    #librosa.display.waveplot(X, sr=s_rate, ax=ax[0])
    #plt.show()
    return X

"""
feature_extraction:
extract features (MFCC, Mel-Spectrogram, Tonnetz, Chromagram) from a waveform
input:  
    -X:         sound signal data
    -s_rate:    sampling rate
output: 
    -feature_norm:  normalized feature array  
"""
def feature_extraction(X, s_rate, means, stds):
    mf = np.mean(librosa.feature.mfcc(y=X, sr=s_rate).T, axis=0)
    try:
        t = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=s_rate).T, axis=0)
    except:
        print(f_name)
    m = np.mean(librosa.feature.melspectrogram(X, sr=s_rate).T, axis=0)
    s = np.abs(librosa.stft(X))
    c = np.mean(librosa.feature.chroma_stft(S=s, sr=s_rate).T, axis=0)
    feature = np.concatenate((m, mf, t, c), axis=0)
    # print("feature:", len(feature))
    # print("mel-scaled Spectrogram:", len(m), m)
    # print("MFCC:", len(mf), mf)
    # print("Tonnetz:", len(t), t)
    # print("Chromagram:", len(c), c)

    # normalize the features
    feature_norm = []
    for i in range(len(feature)):
        f = (feature[i] - means['0'][i]) / stds['0'][i]
        feature_norm.append(f)
    return feature_norm

