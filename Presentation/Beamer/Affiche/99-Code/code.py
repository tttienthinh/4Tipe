import numpy as np

def calcul(activation, X, W):
    # Ajout du biais
    X = np.concatenate((X, np.ones((len(X), 1))), axis=1) 
    # Calcul de la sortie
    z = activation(X@W)
    return z