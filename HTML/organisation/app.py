from flask import Flask, render_template, request
from views.MNIST.imageGen import imageGen
from views.MNIST.predict import predict

app = Flask(__name__)
app.register_blueprint(imageGen)
app.register_blueprint(predict)

@app.route('/')
def index():
    return render_template('main_page.html')

@app.errorhandler(Exception)
def error_page(e):
    return render_template('404.html', erreur=str(e))

if __name__ == "__main__":
    app.run(debug=True)