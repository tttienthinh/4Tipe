import numpy as np
import matplotlib.pyplot as plt
from Model import Model, Layer, LayerOptimizer
from Activation import *
from Loss import *









# Données
train_input  = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
train_output = np.array([
    [0],
    [1],
    [1],
    [0]
])

# Creation du model
model = Model([
        LayerOptimizer(2,  2, lr=10, gamma=0.5, activation=sigmoid, d_activation=d_sigmoid),
        LayerOptimizer(2,  1, lr=10, gamma=0.5, activation=sigmoid, d_activation=d_sigmoid),
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

# Affichage
predicted_output = model.predict(train_input)
for input, output in zip(train_input, predicted_output):
    print(f"Pour l'entrée {input} : {output[0]}")
