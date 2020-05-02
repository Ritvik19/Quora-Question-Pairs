import numpy as np
from flask import Flask, request, jsonify, render_template
from utils import *


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():

    text1, text2 = [str(x) for x in request.form.values()]
    features, vects = generateFeatures(text1, text2)
    prediction = getPredictions(features, vects)

    return render_template('index.html', texts=[text1, text2], prediction_text=prediction)

if __name__ == "__main__":
    app.run(debug=True)