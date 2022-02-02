# https://maharshi-yeluri.medium.com/understanding-and-implementing-speech-recognition-using-hmm-6a4e7666de1
import os

input_folder = "/home/tttienthinh/Documents/Programmation/4Tipe/7-HMM/1-audio"
for dirname in os.listdir(input_folder):
    subfolder = os.path.join(input_folder, dirname)
    label = subfolder[subfolder.rfind('/') + 1:]
    print(label)

import os
import numpy as np
from scipy.io import wavfile 
from hmmlearn import hmm #importing GaussianHMM 
import librosa
from librosa.feature import mfcc

class HMMTrainer(object):
    def __init__(self, model_name='GaussianHMM', n_components=4):
        self.model_name = model_name
        self.n_components = n_components

        self.models = []
        if self.model_name == 'GaussianHMM':
            self.model=hmm.GaussianHMM(n_components=4)
        else:
            print("Please choose GaussianHMM")   
    def train(self, X):
        self.models.append(self.model.fit(X))   
    
    def get_score(self, input_data):
        return self.model.score(input_data)

hmm_models = []
for dirname in os.listdir(input_folder):
    # Get the name of the subfolder 
    subfolder = os.path.join(input_folder, dirname)
    if not os.path.isdir(subfolder): 
        continue
    # Extract the label
    label = subfolder[subfolder.rfind('/') + 1:]
    # Initialize variables
    X = np.array([])
    y_words = []

for filename in [x for x in os.listdir(subfolder) if x.endswith('.wav')][:-1]:
    # Read the input file
    filepath = os.path.join(subfolder, filename)
    sampling_freq, audio = librosa.load(filepath)
    # Extract MFCC features
    mfcc_features = mfcc(sampling_freq, audio)
    # Append to the variable X
    if len(X) == 0:
        X = mfcc_features[:,:15]
    else:
        X = np.append(X, mfcc_features[:,:15], axis=0)
    # Append the label
    y_words.append(label)
print('X.shape =', X.shape)

hmm_trainer = HMMTrainer()
hmm_trainer.train(X)
hmm_models.append((hmm_trainer, label))
hmm_trainer = None

input_files = [
    f'{input_folder}/pineapple/pineapple15.wav',
    f'{input_folder}/orange/orange15.wav',
    f'{input_folder}/apple/apple15.wav',
    f'{input_folder}/kiwi/kiwi15.wav'
]

scores=[]
for item in hmm_models:
    hmm_model, label = item
    score = hmm_model.get_score(mfcc_features)
    scores.append(score)
    index=np.array(scores).argmax()
    # Print the output
    # print("\nTrue:", input_file[input_file.find('/')+1:input_file.rfind('/')])
    print("Predicted:", hmm_models[index][1])


hmm_model, label = hmm_models[0]
score = hmm_model.get_score(mfcc_features)
scores.append(score)
index=np.array(scores).argmax()
# Print the output
# print("\nTrue:", input_file[input_file.find('/')+1:input_file.rfind('/')])
print("Predicted:", hmm_models[index][1])