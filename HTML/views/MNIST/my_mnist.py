#!/usr/bin/env python
# coding: utf-8

# In[210]:


import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import time, pickle, cv2

# # POO

# ## Activation

# In[154]:


"""
Fonction activation Linéaire
"""
def lineaire(x, derive=False):
    if derive:
        np.ones(x.shape)
    return x

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
"""
def softmax(y, derivative=False):
    # Numerically stable with large exponentials
    result = []
    for x in y:
        exps = np.exp(x - x.max())
        if derivative:
            result.append(exps / np.sum(exps, axis=0) * (1 - exps / np.sum(exps, axis=0)))
        else:
            result.append(exps / np.sum(exps, axis=0))
    return np.array(result)

"""
Fonction activation Softmax
"""
def maxi(y, derivative=False):
    # Numerically stable with large exponentials
    result = []
    for x in y:
        exps = np.exp(x - x.max())
        if derivative:
            result.append(exps / np.sum(exps, axis=0) * (1 - exps / np.sum(exps, axis=0)))
        else:
            result.append(exps / np.sum(exps, axis=0))
    return np.array(result)


# ## Layers

# In[155]:




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

# In[156]:


"""
Mean Square Error function
"""
def mse(predicted_output, target_output, derivate=False):
    if derivate:
        return (predicted_output - target_output) *2 
    return ((predicted_output - target_output) ** 2).mean()  


# ## Model

# In[159]:


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
        '''
            This function does a forward pass of x, then checks if the indices
            of the maximum value in the output equals the indices in the label
            y. Then it sums over each prediction and calculates the accuracy.
        '''
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


