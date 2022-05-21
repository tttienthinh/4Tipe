#!/usr/bin/env python3

import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt

data = [
    ['/a/', '"a"', 'sac', [750, 1450, 2600]], 
    ['/∂/', '"e"', 'ce', [550, 1550, 2550]], 
    ['/i/', '"i"', 'lit', [250, 2250, 3000]], 
    ['/o/', '"o"', 'vélo', [350, 750, 2550]], 
    ['/y/', '"u"', 'lune', [250, 1750, 2150]]
]

sampleRate = 16000
length = 3

for api, o, mot, frequencies in data:
    t = np.linspace(0, length, sampleRate * length)  #  Produces a 5 second Audio-File
    mean = 0
    std = 1
    y = np.random.normal(mean, std, size=sampleRate * length)*0.0001
    for frequency in frequencies:
        start = 1
        end = 2
        y[sampleRate*start: sampleRate*end] += np.sin(frequency * 2 * np.pi * t)[sampleRate*start: sampleRate*end] 
    wavfile.write(f'2-data/sound/{o}.wav', sampleRate, y)

    fs, data = wavfile.read('Sine.wav')
    # Filtering and plotting
    # y = butter_lowpass_filter(data[:, 0], cutoff, fs, order)

    powerSpectrum, frequenciesFound, time, imageAxis = plt.specgram(data, Fs=fs, scale_by_freq=True, cmap="gray")
    x1,x2,y1,y2 = plt.axis()  
    plt.axis((x1,x2,0,8_000))
    plt.title(f"Spectrogram : {api}, {o}")
    plt.xlabel('Time [sec]')
    plt.ylabel('Frequency [Hz]')
    plt.savefig(f"2-data/spectre/{o}.jpg")

