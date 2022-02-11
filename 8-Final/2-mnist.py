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

"""
Normal
egalisation = 5_000
y_train = np.zeros((len(Y_train), 10))
y_train[np.arange(len(Y_train)), Y_train] = 1 # to categorical
y_test = np.zeros((len(Y_test), 10))
y_test[np.arange(len(Y_test)), Y_test] = 1 # to categorical 

# cela permet de transformer la sortie en une liste [0, 0, 0, 0, 0, 0, 0, 0 ,0, 0, 0] 
# avec un 1 à l'indice n
# par exemple si le nombre cherché est 2 : [0, 0, 1, 0, 0, 0, 0, 0 ,0, 0, 0] 

x_train = X_train.reshape(-1, 28*28)/255 # 28*28 = 784
x_test = X_test.reshape(-1, 28*28)/255

"""
"""
égalisation
"""
egalisation = 5_000
y_train = np.zeros((10*egalisation, 10))
x_train = np.empty((0, 784))
for i in range(10):
    y_train[np.arange(i*egalisation, (i+1)*egalisation), i] = 1 # to categorical
    x_train = np.concatenate((x_train, X_train[Y_train==i][:egalisation].reshape(-1, 28*28)/255)) # 28*28 = 784


y_test = np.zeros((len(Y_test), 10))
y_test[np.arange(len(Y_test)), Y_test] = 1 # to categorical 
x_test = X_test.reshape(-1, 28*28)/255


""" Méthode Bensal zéro ou non
y_train = np.zeros((len(Y_train), 2))
y_train[np.arange(len(Y_train)), (Y_train!=0)*1] = 1
y_test = np.zeros((len(Y_test), 2))
y_test[np.arange(len(Y_test)), (Y_test!=0)*1] = 1 # to categorical 
"""


# Creation du model
model = ModelClassification([
        LayerOptimizer(784, 32, lr=0.9, gamma=0.5),
        LayerOptimizer(32, 16, lr=0.9, gamma=0.5),
        LayerOptimizer(16, 10, lr=0.9, gamma=0.5, activation=softmax, d_activation=d_softmax),
    ],
    loss_function=cross_entropy,
    d_loss_function=d_cross_entropy
)


# Entrainement
losses = []
accs = []
epochs = 50
for epoch in range(epochs +1):
    y, loss, acc = model.backpropagation(x_train, y_train)
    losses.append(loss)
    accs.append(acc)
    if epoch%10 == 0:
        print(f"Epoch {epoch} : {round(acc*100, 2)}% Accuracy")

plt.plot(losses, label="losses")
plt.plot(accs, label="accs")
plt.legend()
plt.show()

print(model.backpropagation(x_train, y_train)[1:])
print(model.backpropagation(x_test, y_test)[1:])

model.backpropagation(x_test, y_test)[0].argmax(axis=-1)