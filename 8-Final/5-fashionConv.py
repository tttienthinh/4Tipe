from tensorflow.keras.datasets import fashion_mnist # Seulement pour importer les images
import matplotlib.pyplot as plt
import numpy as np
import pickle

from Model import Model, ModelClassification
from Layer import Layer, LayerOptimizer
from Activation import *
from Loss import *
from Convolutional import Convolutional, Flatten
import time

print(time.time())

# Traitement des données
(X_train, Y_train), (X_test, Y_test) = fashion_mnist.load_data()

y_train = np.zeros((len(Y_train), 10))
y_train[np.arange(len(Y_train)), Y_train] = 1 # to categorical
y_test = np.zeros((len(Y_test), 10))
y_test[np.arange(len(Y_test)), Y_test] = 1 # to categorical 

# cela permet de transformer la sortie en une liste [0, 0, 0, 0, 0, 0, 0, 0 ,0, 0, 0] 
# avec un 1 à l'indice n
# par exemple si le nombre cherché est 2 : [0, 0, 1, 0, 0, 0, 0, 0 ,0, 0, 0] 

x_train = X_train.reshape(-1, 1, 28, 28)/255 # 28*28 = 784
x_test = X_test.reshape(-1, 1, 28, 28)/255

"""
Convolutional(img_size=26, kernel_size=3, nb_kernel=4), # 28-kernel_size+1=26, nb_image=3
Flatten(24), # 26-3+1=24, nb_image=9            
"""
# Creation du model
model = ModelClassification([
        # LayerOptimizer(784, 256, lr=0.5, gamma=0.5, activation=sigmoid, d_activation=d_sigmoid),
        Convolutional(img_size=28, kernel_size=3, nb_kernel=4),
        Flatten(26), # 28-kernel_size+1=26, nb_image=3
        LayerOptimizer(2704, 10, lr=0.5, gamma=0.5, activation=softmax, d_activation=d_softmax),
    ],
    loss_function=cross_entropy,
    d_loss_function=d_cross_entropy
)


# Entrainement
losses = []
accs = []
epochs = 100
longueur = len(x_train)
for epoch in range(epochs+1):
    end = (1000*(epoch+1) -1)% longueur+1
    start = max(0, end-100)
    y, loss, acc = model.backpropagation(x_train[:15000], y_train[:15000])
    losses.append(loss)
    accs.append(acc*100)
    time.sleep(10)
    if epoch%5 == 0:
        with open(f'pickle/5-fashionConv-{epoch}.pickle', 'wb') as handle:
           pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Epoch {epoch} : {round(acc*100, 2)}% Accuracy")

"""
with open('5-fashionConv.pickle', 'wb') as handle:
    pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('5-fashionConv.pickle', 'rb') as handle:
    model = pickle.load(handle)

python 5-fashionConv.py > data
sleep 600
shutdown now


"""

# Affichage résultat
fig, axs = plt.subplots(2, 1, figsize=(12, 12))
axs[0].plot(losses)
axs[0].set_title("Courbe d'erreur")
axs[1].plot(accs)
axs[1].set_title("Taux de précision (%)")
axs[1].set_ylim([0, 100])
for i in range(2):
    axs[i].grid()
plt.savefig("Accs.jpg", dpi=400)

dico = [
    "T-shirt",
    "Trouser",
    "Pull",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Boot"
]

fig = plt.figure(figsize=(15,10))
start = 40
end = start + 40
test_preds = model.predict(x_test[start:end]).argmax(axis=-1)
for i in range(40):  
    ax = fig.add_subplot(5, 8, (i+1))
    ax.imshow(X_test[start+i], cmap=plt.get_cmap('gray'))
    if Y_test[start+i] != test_preds[i]:
        ax.set_title('{res}'.format(res=dico[test_preds[i]]), color="red")
    else:
        ax.set_title('{res}'.format(res=dico[test_preds[i]]))
    plt.axis('off')
plt.savefig("Resultat.jpg", dpi=400)

print(time.time())
