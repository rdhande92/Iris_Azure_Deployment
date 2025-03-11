from flask import Flask, jsonify, request
import numpy as np
import joblib

import flask
app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route('/index')
def index():
    return flask.render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        KNN_Model = joblib.load('model.pkl')

        to_predict_list = request.form.to_dict()
        print("Dict of Value from HTML Page - ",to_predict_list)

        to_predict_list = list(to_predict_list.values())
        print("List of Value from HTML Page - ",to_predict_list)

        to_predict_list = np.array(list(map(float, to_predict_list))).reshape(1, -1)
        print("2-D numpy array of Value - ",to_predict_list)


        prediction = KNN_Model.predict(to_predict_list)
        return jsonify({"Model predicted result : ": list(prediction)[0]})

    except Exception as e:
        print("Some Exception:-",e)
        return 'Hello Exception!'



if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8082,debug=True)
