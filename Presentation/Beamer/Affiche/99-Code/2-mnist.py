from tensorflow.keras.datasets import mnist # Seulement pour importer les images
import matplotlib.pyplot as plt
import numpy as np
import pickle

from Model import Model, ModelClassification
from Layer import Layer, LayerOptimizer
from Activation import *
from Loss import *





# Traitement des données
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
y_train = np.zeros((len(Y_train), 10))
y_train[np.arange(len(Y_train)), Y_train] = 1 # to categorical
y_test = np.zeros((len(Y_test), 10))
y_test[np.arange(len(Y_test)), Y_test] = 1 # to categorical 
x_train = X_train.reshape(-1, 28*28)/255 # 28*28 = 784
x_test = X_test.reshape(-1, 28*28)/255

# Creation du model
model = ModelClassification([
        LayerOptimizer(784, 10, lr=0.5, gamma=0.5, activation=softmax, d_activation=d_softmax),
    ],
    loss_function=cross_entropy,
    d_loss_function=d_cross_entropy
)

# Entrainement
losses = []
accs = []
epochs = 100
for epoch in range(epochs):
    y, loss, acc = model.backpropagation(x_train, y_train)
    losses.append(loss) # Permet l'affichage des courbes
    accs.append(acc*100)

# Affichage résultat
predicted_output, loss, acc = model.predict_accuracy(x_test, y_test)
print(f"Après {epochs} générations : le taux de bonnes réponses est {acc*100}%")
