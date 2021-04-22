from flask import Flask, render_template, request, send_from_directory, url_for
from flask import Blueprint

import numpy as np
import json, cv2, os, sys, shutil
from PIL import Image
from zipfile import ZipFile


imageGen = Blueprint('imageGen', __name__, template_folder="templates")
stroke = [2, 4, 8, 16, 32]
nombre = 0
name = 0

@imageGen.route('/MNIST/imageGen')
def imageGen_index():
    return render_template("MNIST/imageGen.html", stroke=stroke, nombre=0)

@imageGen.route('/MNIST/enregistrer/<int:i>', methods=['POST', 'GET'])
def enregistrer(i):
    global nombre, name
    nb = nombre
    nm = name
    if request.method == "POST":
        file_val = request.data
        fig = cv2.imdecode(np.fromstring(file_val, np.uint8), cv2.IMREAD_UNCHANGED) #Convert from string to cv2 image
        img = cv2.cvtColor(fig, cv2.COLOR_RGB2GRAY) # Convert to grayscale
        Image.fromarray(img).save(f'static/image/imageGen/128/{nb}/{stroke[i]}/{nm}.png')
        img = cv2.resize(img, (28, 28), interpolation = cv2.INTER_AREA)
        Image.fromarray(img).save(f'static/image/imageGen/28/{nb}/{stroke[i]}/{nm}.png')
    else:
        print(i)
        print("test")
        nombre += 1
        name += nombre//10
        nombre = nombre%10

    return json.dumps({'nombre': nombre}), 200, {'ContentType': 'application/json'}

@imageGen.route('/MNIST/effacer_fichier', methods=['GET'])
def effacer_fichier():
    global nombre, name
    file_val = request.data
    for pixel in [28, 128]:
        for nb in range(10):
            shutil.rmtree(f'static/image/imageGen/{pixel}/{nb}')
            os.mkdir(f'static/image/imageGen/{pixel}/{nb}')
            for s in stroke:
                os.mkdir(f'static/image/imageGen/{pixel}/{nb}/{s}')
    nombre = 0
    name = 0
    return json.dumps({'nombre': nombre}), 200, {'ContentType': 'application/json'}
                
def create_zip(zipObj, dirName):
    for root, dirs, files in os.walk(dirName):
        root.replace("static/image/", "")
        for file in files:
            zipObj.write(os.path.join(root, file))
        for directory in dirs:
            zipObj.write(os.path.join(root, directory))


@imageGen.route('/MNIST/telecharger', methods=['GET'])
def telecharger():
    print("telecharger")
    # create a ZipFile object
    zipObj = ZipFile("static/image/imageGen.zip", 'w')
    # filling the zipObj
    create_zip(zipObj, "static/image/imageGen")
    # close the Zip File
    zipObj.close()

    return send_from_directory(directory="static/image/", filename='imageGen.zip', as_attachment=True)