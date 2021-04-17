from flask import Flask, render_template, request
from flask import Blueprint
import json

imageGen = Blueprint('imageGen', __name__, template_folder="templates")
stroke = [2, 4, 8, 16, 32]
nombre=0

@imageGen.route('/MNIST/imageGen')
def imageGen_index():
    return render_template("MNIST/imageGen.html", stroke=stroke, nombre=0)

@imageGen.route('/MNIST/enregistrer')
def enregistrer():
    print("enregistrer")
    file_val = request.data

    fig = cv2.imdecode(np.fromstring(request.data, np.uint8), cv2.IMREAD_UNCHANGED) #Convert from string to cv2 image
    img = cv2.cvtColor(fig, cv2.COLOR_RGB2GRAY) # Convert to grayscale
    nombre = int(5)
    return json.dumps({'num': nombre}), 200, {'ContentType': 'application/json'}