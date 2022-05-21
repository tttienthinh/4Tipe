import numpy as np
import matplotlib.pyplot as plt
from Layer import Layer, LayerOptimizer
import Activation
import Loss

"""
17/03/2021 tranthuongtienthinh
Feed Forward model pour résoudre le xor
algorithme POO
"""


class Model:

    def __init__(self, layers=[], loss_function=Loss.mse, d_loss_function=Loss.d_mse):
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
    def __init__(self, layers=[], loss_function=Loss.mse, d_loss_function=Loss.d_mse):
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


