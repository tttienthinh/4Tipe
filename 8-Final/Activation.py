import numpy as np
import matplotlib.pyplot as plt

"""
Linéaire
https://medium.com/@omkar.nallagoni/activation-functions-with-derivative-and-python-code-sigmoid-vs-tanh-vs-relu-44d23915c1f4
"""
def lineaire(x):
    """
    Fonction activation Linéaire
    """
    return x

def d_lineaire(x):
    """
    Dérivé Linéaire
    """
    return np.ones(x.shape)


"""
Sigmoid
"""
def sigmoid(x):
    """
    Fonction Sigmoid
    """
    return 1 / (1 + np.exp(-np.clip(x, -5, 5)))

def d_sigmoid(x):
    """
    Dérivé de Sigmoid
    """
    return x * (1-x)

"""
Tanh
"""
def tanh(x):
    pass

"""
Softmax
"""
def softmax(x):
    exp = np.exp(x-x.max(axis=-1, keepdims=True))
    return exp / exp.sum(axis=-1, keepdims=True)

def d_softmax(x):
    # s = softmax(x)
    # return s * (1 - s)
    return np.ones(x.shape)

"""
https://sgugger.github.io/a-simple-neural-net-in-numpy.html
https://www.adeveloperdiary.com/data-science/deep-learning/neural-network-with-softmax-in-python/
"""