from tensorflow.keras.datasets import mnist # Seulement pour importer les images
import matplotlib.pyplot as plt
import numpy as np
import pickle

from Model import Model, ModelClassification
from Layer import Layer, LayerOptimizer
from Activation import *
from Loss import *

"""
(trainX, trainy), (testX, testy) = fashion_mnist.load_data()
# summarize loaded dataset
print('Train: X=%s, y=%s' % (trainX.shape, trainy.shape))
print('Test: X=%s, y=%s' % (testX.shape, testy.shape))
# plot first few images
for i in range(9):
	# define subplot
	plt.subplot(330 + 1 + i)
	# plot raw pixel data
	plt.imshow(trainX[i], cmap=plt.get_cmap('gray'))
# show the figure
plt.show()
"""
# Traitement des données
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

"""
Normal
"""
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

# Creation du model
model = ModelClassification([
        LayerOptimizer(784, 64, lr=0, gamma=0.3, activation=sigmoid, d_activation=d_sigmoid),
        LayerOptimizer(64, 10, lr=0.9, gamma=0.3, activation=softmax, d_activation=d_softmax),
    ],
    loss_function=cross_entropy,
    d_loss_function=d_cross_entropy
)


# Entrainement
losses = []
accs = []
epochs = 30
for epoch in range(epochs +1):
    y, loss, acc = model.backpropagation(x_train, y_train)
    losses.append(loss)
    accs.append(acc)
    if epoch%5 == 0:
        print(f"Epoch {epoch} : {round(acc*100, 2)}% Accuracy")

# Affichage résultat
plt.plot(losses, label="losses")
plt.plot(accs, label="accs")
plt.legend()
plt.show()

print(model.backpropagation(x_train, y_train)[1:])
print(model.backpropagation(x_test, y_test)[1:])
model.backpropagation(x_test, y_test)[0].argmax(axis=-1)


fig = plt.figure(figsize=(15,10))
start = 40
end = start + 40
test_preds = model.backpropagation(x_test[start:end], y_test[start:end])[0].argmax(axis=-1)
for i in range(40):  
    ax = fig.add_subplot(5, 8, (i+1))
    ax.imshow(X_test[start+i], cmap=plt.get_cmap('gray'))
    if Y_test[start+i] != test_preds[i]:
        ax.set_title('cible: {cible} - res: {res}'.format(cible=Y_test[start+i], res=test_preds[i]), color="red")
    else:
        ax.set_title('cible: {cible} - res: {res}'.format(cible=Y_test[start+i], res=test_preds[i]))
    plt.axis('off')
plt.title("Résultat")
plt.savefig("Resultat.jpg")
plt.show()
