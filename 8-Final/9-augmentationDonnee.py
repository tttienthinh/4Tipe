import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy.io import wavfile










def pitch_manipulate(data, sampling_rate, pitch_factor):
    return librosa.effects.pitch_shift(data.astype(np.float32), sampling_rate, pitch_factor)


def speed_manipulate(data, speed_factor):
    n = len(data)
    augmented_data = librosa.effects.time_stretch(data.astype(np.float32), speed_factor)
    d = len(augmented_data)
    q, r = n//d, n%d
    augmented_data = np.concatenate(
        [augmented_data]*q +
        [augmented_data[-r:]]
    )
    return augmented_data


def noise_manipulation(data, noise_factor=5_000):
    noise = np.random.randn(len(data))
    augmented_data = data + noise_factor * noise
    augmented_data = augmented_data.astype(type(data[0]))
    return augmented_data


def shift_manipulation(data, sampling_rate, shift):
    augmented_data = np.roll(data, shift)
    return augmented_data



path = "/home/tttienthinh/Documents/Programmation/4Tipe/8-Final/3-mot"

# Audio d'origine
fs, data = wavfile.read(f"{path}/incendie/3.wav")
x = np.linspace(0, 3, 3*fs)
gauche = data[:, 0]
droite = data[:, 1]

plt.figure(figsize=(12, 4))
plt.plot(x, gauche)
plt.plot(x, droite)
plt.legend(["Gauche", "Droite"])
plt.title(f"Audio du mot : INCENDIE")
plt.xlabel('Time [sec]')
plt.savefig(f"{path}/audio/7-origine.jpg", bbox_inches='tight', pad_inches=0)
plt.clf()

# Audio pitch
gauche_pitch = pitch_manipulate(gauche, fs, 10)
droite_pitch = pitch_manipulate(droite, fs, 10)

plt.figure(figsize=(12, 4))
plt.plot(x, gauche_pitch)
plt.plot(x, droite_pitch)
plt.legend(["Gauche", "Droite"])
plt.title(f"Variation de ton")
plt.savefig(f"{path}/audio/7-pitch.jpg", bbox_inches='tight', pad_inches=0)
plt.clf()

# Audio speed
gauche_speed = speed_manipulate(gauche, 1.4)
droite_speed = speed_manipulate(droite, 1.4)

plt.figure(figsize=(12, 4))
plt.plot(x, gauche_speed)
plt.plot(x, droite_speed)
plt.legend(["Gauche", "Droite"])
plt.title(f"Variation de vitesse")
plt.savefig(f"{path}/audio/7-speed.jpg", bbox_inches='tight', pad_inches=0)
plt.clf()

# Audio noise
gauche_noise = noise_manipulation(gauche, 0.02)
droite_noise = noise_manipulation(droite, 0.02)

plt.figure(figsize=(12, 4))
plt.plot(x, gauche_noise)
plt.plot(x, droite_noise)
plt.legend(["Gauche", "Droite"])
plt.title(f"Ajout de bruit")
plt.xlabel('Time [sec]')
plt.savefig(f"{path}/audio/7-noise.jpg", bbox_inches='tight', pad_inches=0)
plt.clf()

# Audio shift
gauche_shift = shift_manipulation(gauche, fs, 4*fs)
droite_shift = shift_manipulation(droite, fs, 4*fs)

plt.figure(figsize=(12, 4))
plt.plot(x, gauche_shift)
plt.plot(x, droite_shift)
plt.legend(["Gauche", "Droite"])
plt.title(f"Décalage temporel")
plt.savefig(f"{path}/audio/7-shift.jpg", bbox_inches='tight', pad_inches=0)
plt.clf()




