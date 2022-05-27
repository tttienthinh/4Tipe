import numpy as np
import librosa












def pitch_manipulate(data, sampling_rate, pitch_factor):
    return librosa.effects.pitch_shift(data.astype(np.float32), sampling_rate, pitch_factor)


def speed_manipulate(data, speed_factor):
    return librosa.effects.time_stretch(data.astype(np.float32), speed_factor)


def noise_manipulation(data, noise_factor=5_000):
    noise = np.random.randn(len(data))
    augmented_data = data + noise_factor * noise
    augmented_data = augmented_data.astype(type(data[0]))
    return augmented_data


def shift_manipulation(data, sampling_rate, shift_max):
    shift = np.random.randint(max(sampling_rate * shift_max, 1))
    augmented_data = np.roll(data, shift)
    return augmented_data