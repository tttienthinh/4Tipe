from flask import Flask, render_template, request
from flask import Blueprint
import json

predict = Blueprint('predict', __name__, template_folder="templates")

@predict.route('/MNIST/predict')
def predict_index():
    version = request.args.get("version")
    print(version)
    return render_template("MNIST/predict.html", version=version)

@predict.route('/MNIST/prediction/<version>')
def prediction(version):
    print(f"predict version {version}")
    message = ""
    if version == "Keras":
        nombre = model_keras.predict()
    elif version == "Numpy":
        nombre = model_numpy.predict()
    else:
        message = f"Attention, {version} n'est pas un model valide, je t'ai choisi Keras par défaut. <br>"
        nombre = model_keras.predict()
    message += f"Resultat : {nombre}"
    return json.dumps({'message': message}), 200, {'ContentType': 'application/json'}