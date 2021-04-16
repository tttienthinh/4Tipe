from flask import Flask, render_template, request
import numpy as np
import cv2
import json
from PIL import Image

import numpy as np
import time, pickle

name = 0







app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html", stroke=[2, 4, 8, 16, 32])

name = 0
number = 0
@app.route('/recognize', methods = ['POST'])
def upldfile():
    if request.method == "POST":
        file_val = request.data
        fig = cv2.imdecode(np.fromstring(request.data, np.uint8), cv2.IMREAD_UNCHANGED) #Convert from string to cv2 image
        img = cv2.cvtColor(fig, cv2.COLOR_RGB2GRAY) # Convert to grayscale
        Image.fromarray(img).save(f"128/{number}/{name}.png")
        return json.dumps({'num': f"128/{number}/{name}.png"}), 200, {'ContentType': 'application/json'}

if __name__ == '__main__':
    app.run(debug=True)
