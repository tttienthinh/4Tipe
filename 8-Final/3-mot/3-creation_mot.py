from scipy import signal
from scipy.fft import fftshift
import matplotlib.pyplot as plt
import numpy as np

import sounddevice as sd
import time
from scipy.io import wavfile
"""
https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.spectrogram.html
"""


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
"""
for count in range(2, 4):
    for mot in mots:
        print(f"Record {mot}")
        time.sleep(1)
        print("Go")
        myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
        sd.wait()  # Wait until recording is finished
        # print(type(myrecording))
        wavfile.write(f"/home/tttienthinh/Documents/Programmation/4Tipe/8-Final/3-mot/agathe_florent/{mot}/{count}.wav", fs, myrecording)  # Save as WAV file 

mot = mots[0]
i = 0
for mot in mots:
    for i in range(5):
        fs, data = wavfile.read(f"/home/tttienthinh/Documents/Programmation/4Tipe/8-Final/3-mot/{mot}/{i}.wav")
        powerSpectrum, frequenciesFound, time, imageAxis = plt.specgram(data[:, 0], Fs=fs, scale_by_freq=True)
        powerSpectrum.shape
        # plt.show()

for mot in mots:
    fs, data = wavfile.read(f"/home/tttienthinh/Documents/Programmation/4Tipe/8-Final/3-mot/{mot}/3.wav")

    powerSpectrum, frequenciesFound, time, imageAxis = plt.specgram(data[:, 0], Fs=fs, scale_by_freq=True)
    x1,x2,y1,y2 = plt.axis()  
    plt.axis((x1,x2,0, 4_000))
    plt.title(f"Spectrogram : {mot.upper()}")
    plt.savefig(f"/home/tttienthinh/Documents/Programmation/4Tipe/8-Final/3-mot/{mot}1.jpg")
    # plt.show()
"""
"""

for i, mot in enumerate(mots):
    fs, data = wavfile.read(f"/home/tttienthinh/Documents/Programmation/4Tipe/8-Final/3-mot/{mot}/3.wav")
    # Filtering and plotting
    # y = butter_lowpass_filter(data[:, 0], cutoff, fs, order)
    
    powerSpectrum, frequenciesFound, time, imageAxis = plt.specgram(data[:, 0], Fs=fs, scale_by_freq=True)
    x1,x2,y1,y2 = plt.axis()  
    plt.axis((x1,x2,0,8_000))
    plt.title(f"Spectrogram {i}: {mot.upper()}")
    plt.xlabel('Time [sec]')
    plt.ylabel('Frequency [Hz]')
    plt.savefig(f"/home/tttienthinh/Documents/Programmation/4Tipe/8-Final/3-mot/Test/{i}-{mot.upper()}.jpg", bbox_inches='tight', pad_inches=0)
    # plt.show()
"""
for mot in mots:
    for i in range(5):
        fs, data = wavfile.read(f"/home/tttienthinh/Documents/Programmation/4Tipe/8-Final/3-mot/{mot}/{i}.wav")
        # Filtering and plotting
        # y = butter_lowpass_filter(data[:, 0], cutoff, fs, order)
        
        powerSpectrum, frequenciesFound, time, imageAxis = plt.specgram(data[:, 0], Fs=fs, scale_by_freq=True)
        x1,x2,y1,y2 = plt.axis()  
        plt.axis((x1,x2,0,8_000))
        plt.title(f"Spectrogram du mot : {mot.upper()}")
        plt.xlabel('Time [sec]')
        plt.ylabel('Frequency [Hz]')
        plt.savefig(f"/home/tttienthinh/Documents/Programmation/4Tipe/8-Final/3-mot/{mot}/{i}.jpg", bbox_inches='tight', pad_inches=0)
        # plt.show()
        plt.clf()

fs, data = wavfile.read(f"/home/tttienthinh/Documents/Programmation/4Tipe/8-Final/3-mot/incendie/3.wav")
plt.plot(np.linspace(0, 3, len(data)), data)
plt.legend(["gauche", "droite"])
plt.title("Audio INCENDIE")
plt.xlabel("Temps [s]")
plt.savefig("Audio.jpg", bbox_inches='tight', pad_inches=0)