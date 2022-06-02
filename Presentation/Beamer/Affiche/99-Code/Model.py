import numpy as np
import matplotlib.pyplot as plt
import Activation
import Loss


"""
17/03/2021 tranthuongtienthinh


"""



class Model:
    """
    Le Model est une liste de couche
    il permet d'organiser :
     - le calcul de la prediction
     - l'entrainement de l'ensemble du model 
    """
    def __init__(self, layers:list, loss_function, d_loss_function):
        self.layers = layers
        self.loss = []
        self.lr = 0.1
        self.loss_function = loss_function
        self.d_loss_function = d_loss_function


    def predict(self, input_data:np.ndarray):
        # On calcule la sortie par récurrence 
        predicted_output = input_data # Initialisation
        for layer in self.layers: # Hérédité
            predicted_output_,predicted_output = layer.calculate(
                predicted_output
            )
        return predicted_output # Sortie attendue


    def predict_loss(self, input_data:np.ndarray, target_output:np.ndarray): 
        # Sortie et Erreur
        predicted_output = self.predict(input_data)
        loss = self.loss_function(
            predicted_output, target_output
        )
        return predicted_output, loss


    def backpropagation(self, input_data:np.ndarray, target_output:np.ndarray):
        # Entrainement par récurrence
        predicted_output, loss = self.predict_loss(
            input_data, target_output
        )
        # dérivée
        d_loss = self.d_loss_function(
            predicted_output, target_output
        ) 
        for i in range(len(self.layers)): # Entrainement
            d_loss = self.layers[-i-1].learn(d_loss)
        self.loss.append(loss)
        return predicted_output, loss


class ModelClassification(Model):
    """
    Permet de calculer le taux de bonne réponse
    """
    def __init__(self, layers:list, loss_function, d_loss_function):
        # classe héritée
        super().__init__(layers, loss_function, d_loss_function)
    

    def predict_accuracy(self, input_data:np.ndarray, target_output:np.ndarray):
        predicted_output, loss = self.predict_loss(
            input_data, target_output
        )
        po = predicted_output
        to = target_output
        nb_bonne_rep = (
            po.argmax(axis=-1) == to.argmax(axis=-1)
        ).sum()
        acc = nb_bonne_rep / len(target_output)
        return predicted_output, loss, acc
    

    def backpropagation(self, input_data:np.ndarray, target_output:np.ndarray):
        # Entrainement par récurrence
        predicted_output, loss, acc = self.predict_accuracy(
            input_data, target_output
        )
        # dérivée
        d_loss = self.d_loss_function(
            predicted_output, target_output
        ) 
        for i in range(len(self.layers)): # Entrainement
            d_loss = self.layers[-i-1].learn(d_loss)
        self.loss.append(loss)
        return predicted_output, loss, acc
