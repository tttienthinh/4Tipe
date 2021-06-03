#!/usr/bin/env python
# coding: utf-8

# Nous allons essayer de reconnaitre des nombres écrit à la main grace à un réseau de neuronne.

# In[1]:


import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pickle


# In[2]:


(X_train, Y_train), (X_test, Y_test) = tf.keras.datasets.mnist.load_data()

y_train = np.zeros((len(Y_train), 10))
y_train[np.arange(len(Y_train)), Y_train] = 1 # to categorical
y_test = np.zeros((len(Y_test), 10))
y_test[np.arange(len(Y_test)), Y_test] = 1 # to categorical 
# cela permet de transformer la sortie en une liste [0, 0, 0, 0, 0, 0, 0, 0 ,0, 0, 0] 
# avec un 1 à l'indice n
# par exemple si le nombre cherché est 2 : [0, 0, 1, 0, 0, 0, 0, 0 ,0, 0, 0] 

x_train = X_train.reshape(-1, 28*28)/255 # 28*28 = 784
x_test = X_test.reshape(-1, 28*28)/255


# # POO

# ## Activation

# In[9]:


"""
Fonction activation Sigmoid
"""
def sigmoid(x, derive=False):
    """
    Fonction Sigmoid
    """
    if derive:
        return np.exp(-x) / ((1+np.exp(-x)) ** 2)
    return 1 / (1 + np.exp(-x))

"""
Fonction activation Softmax
https://levelup.gitconnected.com/killer-combo-softmax-and-cross-entropy-5907442f60ba
"""
def softmax(y, derivative=False):
    result = []
    for x in y:
        exps = np.exp(x - x.max()) # permet d'éviter une exponentielle trop grande
        if derivative:
            result.append(exps / np.sum(exps, axis=0) * (1 - exps / np.sum(exps, axis=0)))
        else:
            result.append(exps / np.sum(exps, axis=0))
    return np.array(result)


# ## Layers

# In[4]:


class Layer:
    def __init__(self, input_n=2, output_n=2, lr=0.1, activation=None):
        """
        Crée un layer de n neuronne connecté aux layer de input neuronnes
        """
        # input_n le nombre d'entrée du neuronne
        # output_n le nombre de neuronne de sortie
        self.weight = np.random.randn(input_n, output_n)
        self.input_n = input_n
        self.output_n = output_n
        self.lr = lr # learning rate

        # the name of the layer is 1
        # next one is 2 and previous 0
        self.predicted_output_ = 0
        self.predicted_output  = 0
        self.input_data = 0

        # Fonction d'activation
        self.activation = activation if activation != None else lineaire

    def calculate(self, input_data):
        """
        Calcule la sortie
        """
        self.input_data = input_data
        # self.input_data = np.concatenate((input_data, np.ones((len(input_data), 1))), axis=1)
        y1 = np.dot(self.input_data, self.weight)
        z1 = self.activation(y1)
        self.predicted_output_ = y1
        self.predicted_output = z1
        return y1, z1

    def learn(self, e_2):
        """
        Permet de mettre à jour les weigths
        """
        e1 = e_2 / self.output_n * self.activation(self.predicted_output_, True)
        # e_0 is for the next layer
        # e_0 = np.dot(e1, self.weight.T)
        e_0 = np.dot(e1, self.weight.T)
        dw1 = np.dot(e1.T, self.input_data)
        self.weight -= dw1.T * self.lr
        return e_0


# ## Loss function

# In[5]:


"""
Mean Square Error function
Je l'utilise mais il serait mieux d'utiliser cross entropy normalement
"""
def mse(predicted_output, target_output, derivate=False):
    if derivate:
        return (predicted_output - target_output) *2 
    return ((predicted_output - target_output) ** 2).mean()


# ## Model

# In[6]:


class Model:

    def __init__(self, layers=[], loss_function=None):
        self.layers = layers
        self.loss = []
        self.lr = 0.1
        self.loss_function = loss_function  

    def predict(self, input_data):
        predicted_output = input_data  # y_ is predicted data
        for layer in self.layers:
            predicted_output_, predicted_output = layer.calculate(predicted_output) # output
        return predicted_output

    def predict_loss(self, input_data, target_output):  # target_output is expected data
        predicted_output = self.predict(input_data)  # y_ is predicted data
        loss = self.loss_function(predicted_output, target_output)
        return predicted_output, loss
    
    def compute_accuracy(self, x_val, y_val):
        predictions = []
        for x, y in zip(x_val, y_val):
            output = self.predict([x])
            pred = np.argmax(output[0])
            predictions.append(pred == np.argmax(y))
        return np.mean(predictions)

    def backpropagation(self, input_data, target_output, batch=None):
        n = len(input_data)
        if batch is None:
            batch = n
        step = n//batch
        losses = []
        for i in range(step):
            b_input_data = input_data[::step]
            b_target_output = target_output[::step]
            predicted_output, loss = self.predict_loss(b_input_data, b_target_output)
            d_loss = self.loss_function(predicted_output, b_target_output, True) # dérivé de loss dy_/dy
            # Entrainement des layers
            for i in range(len(self.layers)):
                d_loss = self.layers[-i - 1].learn(d_loss)
            losses.append(loss)
        loss = sum(losses)/len(losses)
        self.loss.append(loss)
        return loss


# In[10]:


# Le model est de taille 784 -> 32 sigmoid -> 16 sigmoid -> 10 softmax
np.random.seed(2) # permet de rendre le programme reproductible
model = Model([
    Layer(784, 32, 0.001, sigmoid),
    Layer(32, 16, 0.001, sigmoid),
    Layer(16, 10, 0.001, softmax),
], mse)


# In[11]:


# Entrainement
for i in range(50):
    loss = model.backpropagation(x_train, y_train)
    acc = model.compute_accuracy(x_test, y_test)
    print(f"Epoch : {i} loss : {loss}, acc : {round(acc*100, 2)} %")
# Sur un des test précédement réalisé, après 3000 entrainement, on obtient 70% d'accuracy


# In[12]:


# sauvegarde du model
pickle.dump( model, open( "demo.p", "wb" ) )


# # Test sur le model

# In[13]:


model = pickle.load( open( "demo.p", "rb" ) )


# In[14]:


# L'accuracy n'est pas le meme selon les nombres
for i in range(10):
    indexer = (Y_test == i)
    acc = model.compute_accuracy(x_test[indexer], y_test[indexer])
    print(f"For {i} accuracy is {round(acc * 100, 2)}")


# In[30]:


plt.imshow(X_test[2], cmap="gray")
plt.show()


# In[16]:


model.predict([x_test[2]]).argmax()

