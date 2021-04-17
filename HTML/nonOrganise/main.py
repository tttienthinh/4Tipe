from flask import Flask, render_template, request
import numpy as np
import cv2
import json
from PIL import Image
import tensorflow as tf

import numpy as np
import time, pickle
# # POO

# ## Activation

# In[85]:


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

# In[86]:




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

# In[87]:


"""
Mean Square Error function
"""
def mse(predicted_output, target_output, derivate=False):
    if derivate:
        return (predicted_output - target_output) *2 
    return ((predicted_output - target_output) ** 2).mean()  


# ## Model

# In[88]:


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

    def backpropagation(self, input_data, target_output):
        predicted_output, loss = self.predict_loss(input_data, target_output)
        d_loss = self.loss_function(predicted_output, target_output, True) # dérivé de loss dy_/dy
        # Entrainement des layers
        for i in range(len(self.layers)):
            d_loss = self.layers[-i - 1].learn(d_loss)
        self.loss.append(loss)
        return predicted_output, loss


# In[128]:
model = pickle.load( open( "model.p", "rb" ) )

# In[121]:
def predire(img):
    img = cv2.resize(img, dsize=(28, 28), interpolation=cv2.INTER_CUBIC)
    Image.fromarray(img).save("image.png")
    img = (255 - img) / 255
    result = model.predict(img.reshape(1, 784))
    print(result)
    return result[0].argmax()

# In[124]:

modelkeras = tf.keras.models.load_model('model_keras')
name = 0
def predire_keras(img):
    global name
    img = cv2.resize(img, dsize=(28, 28), interpolation=cv2.INTER_CUBIC)
    Image.fromarray(img).save(f"{name}.png")
    name += 1
    img = (255 - img) / 255
    # result = model.predict(img.reshape(1, 784))
    result = model.predict(img)
    print("Keras")
    return result[0].argmax()







app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")


@app.route('/recognize', methods = ['POST'])
def upldfile():
    if request.method == "POST":
        file_val = request.data

        fig = cv2.imdecode(np.fromstring(request.data, np.uint8), cv2.IMREAD_UNCHANGED) #Convert from string to cv2 image
        img = cv2.cvtColor(fig, cv2.COLOR_RGB2GRAY) # Convert to grayscale
        number = int(predire_keras(img))
        return json.dumps({'num': number}), 200, {'ContentType': 'application/json'}

if __name__ == '__main__':
    app.run(debug=True)
