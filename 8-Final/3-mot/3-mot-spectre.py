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
mots = ["a", "e", "i", "o", "u"]


for mot in mots:
    print(f"Record {mot}")
    time.sleep(1)
    print("Go")
    myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
    sd.wait()  # Wait until recording is finished
    print(type(myrecording))
    wavfile.write(f"/home/tttienthinh/Documents/Programmation/4Tipe/8-Final/3-mot/{mot}.wav", fs, myrecording)  # Save as WAV file 

for mot in mots:
    fs, data = wavfile.read(f"/home/tttienthinh/Documents/Programmation/4Tipe/8-Final/3-mot/{mot}.wav")

    powerSpectrum, frequenciesFound, time, imageAxis = plt.specgram(data[:, 0], Fs=fs, scale_by_freq=True)
    x1,x2,y1,y2 = plt.axis()  
    # plt.axis((x1,x2,0,2_500))
    plt.title(f"Spectrogram : {mot.upper()}")
    plt.savefig(f"/home/tttienthinh/Documents/Programmation/4Tipe/8-Final/3-mot/{mot}1.jpg")
    # plt.show()


