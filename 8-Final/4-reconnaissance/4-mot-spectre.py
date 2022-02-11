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

import numpy as np
from scipy.signal import butter, lfilter, freqz
import matplotlib.pyplot as plt



def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


# Setting standard filter requirements.
order = 10
cutoff = 1_000
mots = ["u", "i", "o", "e", "a"]
for i, mot in enumerate(mots):
    fs, data = wavfile.read(f"/home/tttienthinh/Documents/Programmation/4Tipe/8-Final/4-reconnaissance/{mot}.wav")
    # Filtering and plotting
    # y = butter_lowpass_filter(data[:, 0], cutoff, fs, order)
    
    powerSpectrum, frequenciesFound, time, imageAxis = plt.specgram(data[:, 0], Fs=fs, scale_by_freq=True)
    x1,x2,y1,y2 = plt.axis()  
    plt.axis((x1,x2,0,8_000))
    plt.title(f"Spectrogram {i}: {mot.upper()}")
    plt.title(f"Spectrogram {i}: ?")
    plt.xlabel('Time [sec]')
    plt.ylabel('Frequency [Hz]')
    plt.savefig(f"/home/tttienthinh/Documents/Programmation/4Tipe/8-Final/4-reconnaissance/Test/{i}.jpg")
    # plt.show()
    
    """
    plt.clf()
    plt.subplot(2, 1, 1)
    plt.plot(data[:, 0], 'b-', label='data')
    plt.xlabel('Time [sec]')
    plt.grid()
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(y, 'g-', linewidth=2, label='filtered data')
    plt.xlabel('Time [sec]')
    plt.grid()
    plt.legend()
    plt.savefig(f"/home/tttienthinh/Documents/Programmation/4Tipe/8-Final/4-reconnaissance/filtre/{mot}-waveform.jpg")
    """


