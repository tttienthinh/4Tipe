import numpy as np
import matplotlib.pyplot as plt
from Model import Model
from Layer import Layer, LayerOptimizer
from Activation import *
from Loss import *








# Données
train_input  = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
train_output = np.array([[0]   , [1]   , [1]   , [0]   ])
# Creation du model
model = Model([
        LayerOptimizer(
            2,  2, lr=0.5, gamma=0.5, bias=False,
            activation=sigmoid, d_activation=d_sigmoid
        ),
        LayerOptimizer(
            2,  1, lr=0.5, gamma=0.5, bias=False,
            activation=sigmoid, d_activation=d_sigmoid
        ),
    ],
    loss_function=mse,
    d_loss_function=d_mse
)
# Entrainement
losses = []
epochs = 300
for epoch in range(epochs):
    y, loss = model.backpropagation(train_input, train_output)
    losses.append(loss) # Permet l'affichage des courbes
plt.plot(losses)
plt.title("Erreur au cours de l'apprentissage")
plt.show()
