"""
 This is a file for code testing (not include for recording, training, and prediction).
"""
######################### get MFCC image and perform a prediction #####################
import pickle
import numpy as np
from python_speech_features import mfcc
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.models import Model

#CLASS_NAME = ["bubbleWrap", "cardboard", "cotton", "foam", "linen", "nylon", "paper", "polyester", "table", "tin"]
CLASS_NAME = ["cotton",  "foam", "nylon", "linen", "polyester", "cardboard", "rest"]
model_path = "../model/mfcc_6texture.sav"

def get_features(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)

    print("x.shape", x.shape)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    flatten = model.predict(x)
    return list(flatten[0])

"""
# define feature extractor
base_model = VGG19(weights='imagenet')
model = Model(inputs=base_model.input, outputs=base_model.get_layer('flatten').output)

trained_model = pickle.load(open(model_path, 'rb'))  # load trained model
vector = get_features("../data00001_cotton.png", model)

#print(len(vector), vector)
predicted = trained_model.predict([vector])
print(predicted)
"""

from scipy.io.wavfile import read
import numpy as np
from numpy import*
import matplotlib.pyplot as plt

def manipulate(data, sampling_rate, shift_max):
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

input_data = read("../dataset/6texture/data/cotton/data00001_cotton.wav")
plt.figure('1')
audio = input_data[1]
plt.plot(audio)

print(len(audio))
#input_data_roll = np.roll(input_data, int(68000/2))
audio_roll = manipulate(audio, 44100, 0.4)
plt.figure('2')
plt.plot(audio_roll)
plt.show()