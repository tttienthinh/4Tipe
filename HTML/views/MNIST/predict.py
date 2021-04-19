from flask import Flask, render_template, request
from flask import Blueprint

# from .my_mnist import Layer, Model
import tensorflow as tf
import numpy as np
import json, cv2, os, sys, pickle

predict = Blueprint('predict', __name__, template_folder="templates")

@predict.route('/MNIST/predict')
def predict_index():
    version = request.args.get("version")
    print(version)
    return render_template("MNIST/predict.html", version=version)

model_keras = tf.keras.models.load_model('static/models/MNIST/keras_mnist2')
# model_numpy = pickle.load( open( "static/models/MNIST/keras_numpy.p", "rb" ) )



@predict.route('/MNIST/prediction/<version>', methods=["POST"])
def prediction(version):
    print(f"predict version {version}")

    file_val = request.data
    fig = cv2.imdecode(np.fromstring(file_val, np.uint8), cv2.IMREAD_UNCHANGED) #Convert from string to cv2 image
    img = cv2.cvtColor(fig, cv2.COLOR_RGB2GRAY) # Convert to grayscale
    img = cv2.resize(img, (28, 28), interpolation = cv2.INTER_AREA)
    img = (255-img)/255

    message = ""
    if version == "Keras":
        nombre = model_keras.predict(img.reshape(1, 28, 28, 1))[0].argmax()
    elif version == "Numpy":
        nombre = model_keras.predict(img.reshape(1, 28, 28, 1))[0].argmax()
    else:
        message = f"Attention, {version} n'est pas un model valide, je t'ai choisi Keras par défaut. <br>"
        nombre = model_keras.predict()
    message += f"Resultat : {nombre}"
    return json.dumps({'message': message}), 200, {'ContentType': 'application/json'}