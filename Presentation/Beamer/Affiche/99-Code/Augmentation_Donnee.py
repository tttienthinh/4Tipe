import numpy as np
import librosa












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


def noise_manipulation(data, noise_factor):
    noise = np.random.randn(len(data))
    augmented_data = data + noise_factor * noise
    augmented_data = augmented_data.astype(type(data[0]))
    return augmented_data


def shift_manipulation(data, shift):
    augmented_data = np.roll(data, shift)
    return augmented_data