import numpy as np
import matplotlib.pyplot as plt
from Layer import Layer
from Activation import *
from Loss import *

"""
17/03/2021 tranthuongtienthinh
Feed Forward model pour résoudre le xor
algorithme POO
"""


class Model:

    def __init__(self, layers=[], loss_function=None, d_loss_function=None):
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


if __name__ == "__main__":
    # training data XOR !!!
    train_input = np.array([
        [0, 0], [1, 1],
        [0, 1], [1, 0]
    ])
    train_output = np.array([0, 0, 1, 1]).reshape((4, 1))

    # model testing
    l1 = Layer(input_n=2, output_n=2)
    l2 = Layer(input_n=2, output_n=2)
    """
    np.random.seed(2)
    l1.weight = np.random.rand(2, 2)
    l2.weight = np.random.rand(2, 1)
    """
    model = Model([
        l1,
        l2,
    ], loss_function=mse, d_loss_function=d_mse)
    losses = []
    for i in range(100_000):
        y, loss = model.backpropagation(train_input, train_output)
        losses.append(loss)
    plt.plot(losses)
    plt.show()
    print(model.predict(train_input))

