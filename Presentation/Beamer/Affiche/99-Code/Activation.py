import numpy as np

# Linéaire
def lineaire(x):
    return x
def d_lineaire(x):
    return np.ones(x.shape)


# Sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -10, 10)))
def d_sigmoid(x):
    f_x = sigmoid(x)
    return f_x * (1-f_x)


# Tanh
def tanh(x):
    return np.tanh(x)
def d_tanh(x):
    return 1-tanh(x)**2


# ReLU
def relu(x):
    return np.clip(x, 0, np.inf)
def d_relu(x):
    return 1*(x>0)


# Softmax
def softmax(x):
    exp = np.exp(x-x.max(axis=-1, keepdims=True))
    return exp / exp.sum(axis=-1, keepdims=True)
def d_softmax(x):
    return np.ones(x.shape)

