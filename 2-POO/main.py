import numpy as np
import matplotlib.pyplot as plt
from Model import Model, Layer

folder = "video/"
n_image = 0
# affichage
def affichage2d(dataVert, dataBleu, dataRouge, dataOrange, epoch=0, finish=False):
    """
    Cette fonction a été crée par LUCAS RIGOT, modifié par 4T
    """
    global n_image
    plt.scatter(dataVert[0], dataVert[1], color="green", label="Vert = bas gauche")
    plt.scatter(dataBleu[0], dataBleu[1], color="blue", label="Bleu = bas droite")
    plt.scatter(dataRouge[0], dataRouge[1], color="red", label="Rouge = haut gauche")
    plt.scatter(dataOrange[0], dataOrange[1], color="orange", label="Orange = haut droite")
    plt.plot([0, 1], [0.5, 0.5], color="blue")
    plt.plot([0.5, 0.5], [0, 1], color="blue")
    plt.title(f"We are at epoch : {epoch}")
    plt.legend()
    # version arret
    # plt.savefig(f"{folder}/{'%.3d' % n_image}.png")
    n_image += 1
    if finish:
        plt.show()
    else:
        # version pause 1 seconde
        plt.pause(.00_000_000_1)
        plt.clf() # efface l'ancien graphique


# data Creation
train_input = np.random.random((200, 2))
train_output = []
for i in range(200):
    data = train_input[i]
    if data[0] < .5 and data[1] < .5:
        train_output.append([1, 0, 0, 0])
    elif data[0] > .5 and data[1] < .5:
        train_output.append([0, 1, 0, 0])
    elif data[0] < .5 and data[1] > .5:
        train_output.append([0, 0, 1, 0])
    else:
        train_output.append([0, 0, 0, 1])
train_output = np.array(train_output)

# model creation
model = Model([
    Layer(2, 8),
    Layer(8, 8),
    Layer(8, 4),
])

# Entrainement
losses = []
epochs = 5_000
for epoch in range(epochs +1):
    y, loss = model.backpropagation(train_input, train_output)
    losses.append(loss)
    if True and epoch % (int(epochs/100)) == 0:
        result = np.argmax(y, axis=1)
        dataVert = (train_input[result == 0]).T
        dataBleu = (train_input[result == 1]).T
        dataRouge = (train_input[result == 2]).T
        dataOrange = (train_input[result == 3]).T
        affichage2d(dataVert, dataBleu, dataRouge, dataOrange, epoch=epoch)

plt.plot(losses)
# plt.savefig(f"{folder}/{'%.3d' % n_image}.png")
plt.show()


