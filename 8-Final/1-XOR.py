import numpy as np
import matplotlib.pyplot as plt
from Model import Model, Layer, LayerOptimizer
import Activation, Loss

def affichage2d(dataVert, dataBleu, epoch=0, finish=False):
    """
    Cette fonction a été crée par LUCAS RIGOT, modifié par 4T
    """
    global n_image
    plt.scatter(dataVert[0], dataVert[1], color="green", label="Vert = bas gauche")
    plt.scatter(dataBleu[0], dataBleu[1], color="blue", label="Bleu = haut droite")
    plt.title(f"Epoch : {epoch}")
    if finish:
        plt.show()
    else:
        # version pause 1 seconde
        plt.pause(0.000_1)
        plt.clf() # efface l'ancien graphique

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

# model creation
model = Model([
        LayerOptimizer(2,  2, lr=10, gamma=0.5, activation=Activation.sigmoid, d_activation=Activation.d_sigmoid, biais=False, mini=0, maxi=1),
        LayerOptimizer(2,  1, lr=10, gamma=0.5, activation=Activation.sigmoid, d_activation=Activation.d_sigmoid, biais=False, mini=0, maxi=1),
    ],
    loss_function=Loss.mse,
    d_loss_function=Loss.d_mse
)
# Entrainement
losses = []
epochs = 300

for epoch in range(epochs +1):
    y, loss = model.backpropagation(train_input, train_output)
    losses.append(loss)


    if True and epoch % (int(epochs/100)) == 0:
        dataVert = (train_input[(y < 0.5)[:, 0]]).T
        dataBleu = (train_input[(y > 0.5)[:, 0]]).T
        affichage2d(dataVert, dataBleu, epoch=epoch)


print(len(train_input))
plt.plot(losses)
plt.title("XOR : Erreur au cours des générations")
plt.savefig(f"XOR.png", dpi=400)
for i in range(2):
    print(model.layers[i].weight)

print("Resultat")
print(model.predict(train_input))
plt.show()

"""
[[0.85101517 5.42276618]
 [0.8507147  5.40598544]
 [0.14255344 0.44279295]]
[[-18.39232774]
 [ 14.42960901]
 [  0.02451323]]
Resultat
[[0.121174  ]
 [0.81416403]
 [0.81415036]
 [0.24479818]]

model = Model([
        LayerOptimizer(2,  2, lr=10, gamma=0.5, activation=Activation.sigmoid, d_activation=Activation.d_sigmoid, biais=False, mini=0, maxi=1),
        LayerOptimizer(2,  1, lr=10, gamma=0.5, activation=Activation.sigmoid, d_activation=Activation.d_sigmoid, biais=False, mini=0, maxi=1),
    ],
    loss_function=Loss.mse,
    d_loss_function=Loss.d_mse
)
model.layers[0].weight = np.array([[0.85101517, 5.42276618],
 [0.8507147,  5.40598544],
 [0.14255344 ,0.44279295]]
 )

model.layers[1].weight = np.array([[-18.39232774],
 [ 14.42960901],
 [  0.02451323]])

print(model.predict_loss(train_input, train_output))
"""

