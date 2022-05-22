from tensorflow.keras.datasets import mnist # Seulement pour importer les images
import numpy as np
import matplotlib.pyplot as plt


"""
Cross-Entropy Loss
https://levelup.gitconnected.com/killer-combo-softmax-and-cross-entropy-5907442f60ba#aee4
https://gombru.github.io/2018/05/23/cross_entropy_loss#losses
https://www.tensorflow.org/api_docs/python/tf/keras/metrics/categorical_crossentropy
"""
def cross_entropy(predicted_output, target_output):
    return -(target_output * np.log(np.clip(np.abs(predicted_output), 1e-3, 1))).sum(axis=-1).mean()

def d_cross_entropy(predicted_output, target_output):
    return predicted_output - target_output
"""
https://youtu.be/xBEh66V9gZo?t=973
"""

"""
https://sgugger.github.io/a-simple-neural-net-in-numpy.html#cross-entropy-cost
"""

"""
Sigmoid
"""
def sigmoid(x):
    """
    Fonction Sigmoid
    """
    return 1 / (1 + np.exp(-np.clip(x, -10, 10)))

def d_sigmoid(x):
    """
    Dérivé de Sigmoid
    """
    f_x = sigmoid(x)
    # f_x = x
    return f_x * (1-f_x)


def softmax(x):
    exp = np.exp(x-x.max(axis=-1, keepdims=True))
    return exp / exp.sum(axis=-1, keepdims=True)

def d_softmax(x):
    # s = softmax(x)
    # return s * (1 - s)
    return np.ones(x.shape)

class Layer:
    def __init__(self, input_n=2, output_n=2, lr=0.1, activation=sigmoid, d_activation=d_sigmoid, biais=True, mini=0, maxi=1):
        """
        Crée un layer de n neuronne connecté aux layer de input neuronnes
        """
        # input_n le nombre d'entrée du neuronne
        # output_n le nombre de neuronne de sortie
        self.weight = np.random.rand(input_n+1, output_n)*(maxi-mini)+mini
        self.biais = biais
        self.input_n = input_n
        self.output_n = output_n
        self.lr = lr # learning rate

        # the name of the layer is 1
        # next one is 2 and previous 0
        self.predicted_output_ = 0
        self.predicted_output  = 0
        self.input_data = 0

        # Fonction d'activation
        self.activation = activation
        self.d_activation = d_activation

    def next(self):
        return self.output_n

    def calculate(self, input_data):
        """
        Calcule la sortie
        """
        # Ajout du biais
        if self.biais:
            self.input_data = np.concatenate((input_data, np.ones((len(input_data), 1))), axis=1) 
        else:
            self.input_data = np.concatenate((input_data, np.zeros((len(input_data), 1))), axis=1) 
        y1 = np.dot(self.input_data, self.weight)
        z1 = self.activation(y1)
        self.predicted_output_ = y1
        self.predicted_output = z1
        return y1, z1

    def learn(self, e_2):
        """
        Permet de mettre à jour les poids weigth
        """
        e1 = e_2 / (self.input_n+1) * self.d_activation(self.predicted_output)
        # e_0 est pour l'entrainement de la couche précédente
        e_0 = np.dot(e1, self.weight.T)[:, :-1]
        dw1 = np.dot(e1.T, self.input_data)
        self.weight -= dw1.T * self.lr
        return e_0



class Model:

    def __init__(self, layers, loss_function, d_loss_function):
        self.layers = layers
        self.loss = []
        self.lr = 0.1
        self.loss_function = loss_function
        self.d_loss_function = d_loss_function

    def predict(self, input_data):
        predicted_output = input_data  # y_ is predicted data
        for layer in self.layers:
            predicted_output_, predicted_output = layer.calculate(predicted_output) # output
        return predicted_output

    def predict_loss(self, input_data, target_output):  # target_output is expected data
        predicted_output = self.predict(input_data)  # y_ is predicted data
        loss = self.loss_function(predicted_output, target_output)
        return predicted_output, loss

    def backpropagation(self, input_data, target_output):
        predicted_output, loss = self.predict_loss(input_data, target_output)
        d_loss = self.d_loss_function(predicted_output, target_output) # dérivé de loss dy_/dy
        # Entrainement des layers
        for i in range(len(self.layers)):
            d_loss = self.layers[-i - 1].learn(d_loss)
        self.loss.append(loss)
        return predicted_output, loss

class ModelClassification(Model):
    def __init__(self, layers   , loss_function, d_loss_function):
        super().__init__(layers, loss_function, d_loss_function)
    
    def predict_accuracy(self, input_data, target_output):
        predicted_output, loss = self.predict_loss(input_data, target_output)
        nb_bonne_rep = (predicted_output.argmax(axis=-1) == target_output.argmax(axis=-1)).sum()
        acc = nb_bonne_rep / len(target_output)
        return predicted_output, loss, acc
    
    def backpropagation(self, input_data, target_output):
        predicted_output, loss, acc = self.predict_accuracy(input_data, target_output)
        d_loss = self.d_loss_function(predicted_output, target_output) # dérivé de loss dy_/dy
        # Entrainement des layers
        for i in range(len(self.layers)):
            d_loss = self.layers[-i - 1].learn(d_loss)
        self.loss.append(loss)
        return predicted_output, loss, acc



from tensorflow.keras.datasets import mnist # Seulement pour importer les images
import matplotlib.pyplot as plt
import numpy as np

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

# cela permet de transformer la sortie en une liste [0, 0, 0, 0, 0, 0, 0, 0 ,0, 0, 0] 
# avec un 1 à l'indice n
# par exemple si le nombre cherché est 2 : [0, 0, 1, 0, 0, 0, 0, 0 ,0, 0, 0] 

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
epochs = 50
for epoch in range(epochs):
    y, loss, acc = model.backpropagation(x_train, y_train)
    losses.append(loss)
    accs.append(acc*100)
    if epoch%5 == 0:
        print(f"Epoch {epoch} : {round(acc*100, 2)}% Accuracy")

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

print(model.backpropagation(x_train, y_train)[1:])
print(model.backpropagation(x_test, y_test)[1:])
model.backpropagation(x_test, y_test)[0].argmax(axis=-1)


fig = plt.figure(figsize=(15,10))
start = 40
end = start + 40
test_preds = model.predict(x_test[start:end]).argmax(axis=-1)
for i in range(40):  
    ax = fig.add_subplot(5, 8, (i+1))
    ax.imshow(X_test[start+i], cmap=plt.get_cmap('gray'))
    if Y_test[start+i] != test_preds[i]:
        ax.set_title('Prediction: {res}'.format(res=test_preds[i]), color="red")
    else:
        ax.set_title('Prediction: {res}'.format(res=test_preds[i]))
    plt.axis('off')
plt.savefig("Resultat.jpg", dpi=400)
