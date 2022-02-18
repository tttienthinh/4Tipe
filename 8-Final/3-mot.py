# from tensorflow.keras.datasets import mnist # Seulement pour importer les images
import matplotlib.pyplot as plt
import numpy as np
import pickle

from Model import Model, ModelClassification
from Layer import Layer, LayerOptimizer
from Activation import *
from Loss import *

from scipy import signal
from scipy.fft import fftshift
import matplotlib.pyplot as plt
import numpy as np

import sounddevice as sd
import time
from scipy.io import wavfile
import librosa

def noise_manipulation(data, noise_factor=5_000):
    noise = np.random.randn(len(data))
    augmented_data = data + noise_factor * noise
    # Cast back to same data type
    augmented_data = augmented_data.astype(type(data[0]))
    return augmented_data

def pitch_manipulate(data, sampling_rate, pitch_factor):
    return librosa.effects.pitch_shift(data.astype(np.float32), sampling_rate, pitch_factor)

def speed_manipulate(data, speed_factor):
    return librosa.effects.time_stretch(data.astype(np.float32), speed_factor)

fs = 16000  # Sample rate
seconds = 3  # Duration of recording
mots = [
    "malaise",
    "hémorragie",
    "brûlure",
    "intoxication",
    "violences",
    "agression",
    "vol",
    "cambriolage",
    "incendie",
    "gaz",
    "effondrement",
    "électrocution"
]
output_n = len(mots) # 12

# Traitement des données
(X_train, Y_train), (X_test, Y_test) = ([], []), ([], [])

def augmented(audio):
    L = [audio]
    for i in range(3):
        L.append(noise_manipulation(audio, noise_factor=1_000))
    for i in range(3):
        L.append(pitch_manipulate(audio, fs, 2.5))
    return L


for i_mot, mot in enumerate(mots):
    for i in range(5):
        fs, data = wavfile.read(f"/home/tttienthinh/Documents/Programmation/4Tipe/8-Final/3-mot/{mot}/{i}.wav")
        for audio in augmented(data[:, 0]):
            powerSpectrum, frequenciesFound, time_data, imageAxis = plt.specgram(audio, Fs=fs, scale_by_freq=True)
            powerSpectrum.shape
            
            X_train.append(powerSpectrum) 
            Y_train.append(i_mot)                                                       
"""
for i_mot, mot in enumerate(mots):
    for i in [1, 3]:
        fs, data = wavfile.read(f"/home/tttienthinh/Documents/Programmation/4Tipe/8-Final/3-mot/agathe_florent/{mot}/{i}.wav")
        powerSpectrum, frequenciesFound, time_data, imageAxis = plt.specgram(data[:, 0], Fs=fs, scale_by_freq=True)
        powerSpectrum.shape
        
        X_train.append(powerSpectrum) 
        Y_train.append(i_mot)"""

for i_mot, mot in enumerate(mots):
    for i in range(4):
        fs, data = wavfile.read(f"/home/tttienthinh/Documents/Programmation/4Tipe/8-Final/3-mot/agathe_florent/{mot}/{i}.wav")
        powerSpectrum, frequenciesFound, time_data, imageAxis = plt.specgram(data[:, 0], Fs=fs, scale_by_freq=True)
        powerSpectrum.shape
        
        X_test.append(powerSpectrum) 
        Y_test.append(i_mot)

X_train = np.array(X_train)
X_test = np.array(X_test)

y_train = np.zeros((len(Y_train), output_n))
y_train[np.arange(len(Y_train)), Y_train] = 1 # to categorical
y_test = np.zeros((len(Y_test), output_n))
y_test[np.arange(len(Y_test)), Y_test] = 1 # to categorical 

# cela permet de transformer la sortie en une liste [0, 0, 0, 0, 0, 0, 0, 0 ,0, 0, 0] 
# avec un 1 à l'indice n
# par exemple si le nombre cherché est 2 : [0, 0, 1, 0, 0, 0, 0, 0 ,0, 0, 0] 

x_train = X_train.reshape(-1, 129*374)*1_000 # 129*374 = 48_246
x_test = X_test.reshape(-1, 129*374)*1_000

# Creation du model
model = ModelClassification([
        LayerOptimizer(48_246, 256, lr=0.1, gamma=0.05, activation=sigmoid, d_activation=d_sigmoid),
        LayerOptimizer(256, 256, lr=0.1, gamma=0.05, activation=sigmoid, d_activation=d_sigmoid),
        LayerOptimizer(256, 256, lr=0.1, gamma=0.05, activation=sigmoid, d_activation=d_sigmoid),
        LayerOptimizer(256, 256, lr=0.1, gamma=0.05, activation=sigmoid, d_activation=d_sigmoid),
        LayerOptimizer(256, 64, lr=0.1, gamma=0.05, activation=sigmoid, d_activation=d_sigmoid),
        LayerOptimizer(64, output_n, lr=0.1, gamma=0.05, activation=softmax, d_activation=d_softmax),
    ],
    loss_function=cross_entropy,
    d_loss_function=d_cross_entropy
)


# Entrainement
losses = []
accs = []
epochs = 100
for epoch in range(epochs +1):
    y, loss, acc = model.backpropagation(x_train, y_train)
    losses.append(loss)
    accs.append(acc)
    if epoch%5 == 0:
        print(f"Epoch {epoch} : {round(acc*100, 2)}% Accuracy")



print(f"{round(model.backpropagation(x_test, y_test)[2]*100, 2)}% Accuracy")
"""
# Affichage résultat
plt.plot(losses, label="losses")
plt.plot(accs, label="accs")
plt.legend()
plt.show()

print(model.backpropagation(x_train, y_train)[1:])
print(model.backpropagation(x_test, y_test)[1:])
model.backpropagation(x_test, y_test)[0].argmax(axis=-1)


fig = plt.figure(figsize=(15,10))
start = 40
end = start + 40
test_preds = model.backpropagation(x_test[start:end], y_test[start:end])[0].argmax(axis=-1)
for i in range(40):  
    ax = fig.add_subplot(5, 8, (i+1))
    ax.imshow(X_test[start+i], cmap=plt.get_cmap('gray'))
    if Y_test[start+i] != test_preds[i]:
        ax.set_title('cible: {cible} - res: {res}'.format(cible=Y_test[start+i], res=test_preds[i]), color="red")
    else:
        ax.set_title('cible: {cible} - res: {res}'.format(cible=Y_test[start+i], res=test_preds[i]))
    plt.axis('off')
plt.title("Résultat")
plt.savefig("Resultat.jpg")
plt.show()
"""