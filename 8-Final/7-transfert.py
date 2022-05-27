import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
import librosa

model = tf.keras.models.load_model('model.h5')
model.trainable = False
model3 = tf.keras.Sequential(
    [tf.keras.layers.Input(shape=(374, 129, 1))] 
    + model.layers[0:-1] 
    + [tf.keras.layers.Dense(12, activation="softmax", name="adaptation")]
)

liste = [
    'malaise', 'hémorragie', 'brûlure', 'intoxication', 
    'violences', 'agression', 'vol', 'cambriolage', 
    'incendie', 'gaz', 'effondrement', 'électrocution'
]
dico = {liste[i]:i for i in range(12)}

fs = 16000  # Sample rate
seconds = 3  # Duration of recording
num_labels = 12

def get_spectrogram(waveform):
    # Zero-padding for an audio waveform with less than 16,000 samples.
    input_len = fs*seconds
    waveform = waveform[:input_len]
    zero_padding = tf.zeros(
        [input_len] - tf.shape(waveform),
        dtype=tf.float32)
    # Cast the waveform tensors' dtype to float32.
    waveform = tf.cast(waveform, dtype=tf.float32)
    # Concatenate the waveform with `zero_padding`, which ensures all audio
    # clips are of the same length.
    equal_length = tf.concat([waveform, zero_padding], 0)
    # Convert the waveform to a spectrogram via a STFT.
    spectrogram = tf.signal.stft(
        equal_length, frame_length=255, frame_step=128)
    # Obtain the magnitude of the STFT.
    spectrogram = tf.abs(spectrogram)
    # Add a `channels` dimension, so that the spectrogram can be used
    # as image-like input data with convolution layers (which expect
    # shape (`batch_size`, `height`, `width`, `channels`).
    spectrogram = spectrogram[..., tf.newaxis]
    return spectrogram

def pitch_manipulate(data, sampling_rate, pitch_factor):
    return librosa.effects.pitch_shift(data.astype(np.float32), sampling_rate, pitch_factor)

def speed_manipulate(data, speed_factor):
    return librosa.effects.time_stretch(data.astype(np.float32), speed_factor)

def noise_manipulation(data, noise_factor=5_000):
    noise = np.random.randn(len(data))
    augmented_data = data + noise_factor * noise
    augmented_data = augmented_data.astype(type(data[0]))
    return augmented_data

def shift_manipulation(data, sampling_rate, shift_max, shift_direction):
    shift = np.random.randint(max(sampling_rate * shift_max, 1))
    if shift_direction == 'right':
        shift = -shift
    elif shift_direction == 'both':
        direction = np.random.randint(0, 2)
        if direction == 1:
            shift = -shift
    augmented_data = np.roll(data, shift)
    n = len(data)
    cut = int((n - sampling_rate)/2)
    data_final = augmented_data[cut:-cut] # permet d'obtenir des son de 1 sec 48kbit
    return data_final[:sampling_rate]

def augmentation(audio):
    audios = [audio]
    new_audios = []
    for _ in range(1): # 3+1 fois plus de données
        for audio in audios:
            pitch = np.random.random() - 0.5 # ptich varie entre [-0.5 et 0.5]
            new_audios.append(pitch_manipulate(audio, 16_000, pitch))
    audios += new_audios.copy()

    new_audios = [] # SPEED
    for i in range(1): # 3+1 fois plus de données
        for audio in audios:
            speed = np.random.random() + 0.5 # speed varie entre [0.5 et 1.5]
            new_audios.append(speed_manipulate(audio, speed))
    audios += new_audios.copy()

    new_audios = [] # NOISE
    for noise_max in [0.01, 0.1]: # 3+1 fois plus de données
        for audio in audios:
            noise = np.random.random() * noise_max  
            tmp = noise_manipulation(audio, noise_factor=noise)
            new_audios.append(tmp / max(tmp))
    audios += new_audios.copy()

    new_audios = [] # SHIFT
    for shift in range(1, 2): # 4 fois plus de données
        for audio in audios:
            new_audios.append(shift_manipulation(audio, 16_000, shift/4, "both")) # shift varie dans [0, 0.25, 0.5, 0.75]
    audios = new_audios.copy()
    return audios

x_train, x_test = [], []
Y_train, Y_test = [], []
for i in range(5):
    for mot in liste:
        fs, data = wavfile.read(f"/home/tttienthinh/Documents/Programmation/4Tipe/8-Final/3-mot/{mot}/{i}.wav")
        if i < 4:
            audios = [data[:, 0]]
            # audios = augmentation(data[:, 0])
            x_train += [get_spectrogram(audio) for audio in audios]
            Y_train += len(audios)*[dico[mot]]
        else:
            spectrogram = get_spectrogram(data[:, 0])
            x_test.append(spectrogram)
            Y_test.append(dico[mot])

for i in range(4):
    for mot in liste:
        fs, data = wavfile.read(f"/home/tttienthinh/Documents/Programmation/4Tipe/8-Final/3-mot/agathe_florent/{mot}/{i}.wav")
        if i in [0, 2]:
            audios = [data[:, 0]]
            # audios = augmentation(data[:, 0])
            x_train += [get_spectrogram(audio) for audio in audios]
            Y_train += len(audios)*[dico[mot]]
        else:
            spectrogram = get_spectrogram(data[:, 0])
            x_test.append(spectrogram)
            Y_test.append(dico[mot])


x_train = np.array(x_train)
x_test = np.array(x_test)

y_train = np.zeros((len(Y_train), num_labels))
y_test = np.zeros((len(Y_test), num_labels))
y_train[np.arange(len(Y_train)), Y_train] = 1 # to categorical
y_test[np.arange(len(Y_test)), Y_test] = 1 # to categorical 

y_test = np.array(Y_test)
y_train = np.array(Y_train)

print(f"Train {len(y_train)}")
print(f"Test {len(y_test)}")
x_train_reshaped = x_train.reshape(-1, 374, 129, 1)
x_test_reshaped = x_test.reshape(-1, 374, 129, 1)

model3.compile(
    loss="sparse_categorical_crossentropy",
    # loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
    optimizer="adam",
    metrics=["accuracy"]
)
EPOCHS = 250
history = model3.fit(
    x_train_reshaped, y_train, 
    validation_data=(x_test_reshaped, y_test),
    # validation_data=val_ds,  
    epochs=EPOCHS,
    # callbacks=tf.keras.callbacks.EarlyStopping(verbose=1, patience=2),
)



# Affichage
metrics = history.history
plt.plot(history.epoch, metrics['loss'], metrics['val_loss'])
plt.legend(['loss', 'val_loss'])
plt.title("Erreur d'apprentissage par transfert")
plt.savefig("7-loss.jpg", bbox_inches='tight', pad_inches=0)
plt.clf()

plt.plot(history.epoch, metrics['accuracy'], metrics['val_accuracy'])
plt.legend(['accuracy', 'val_accuracy'])
plt.title("Précision d'apprentissage par transfert")
plt.savefig("7-accuracy.jpg", bbox_inches='tight', pad_inches=0)
plt.clf()

# test
(model3.predict(x_test_reshaped[:12]).argmax(axis=1) == y_test[:12]).sum()/12
(model3.predict(x_test_reshaped[12:24]).argmax(axis=1) == y_test[12:24]).sum()/12
(model3.predict(x_test_reshaped[24:36]).argmax(axis=1) == y_test[24:36]).sum()/12

(model3.predict(x_test_reshaped).argmax(axis=1) == y_test).sum() / 36

