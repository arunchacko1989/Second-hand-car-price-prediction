import pickle
from flask import Flask, render_template, request, app, jsonify, url_for
import numpy
import pandas as pd
from sklearn.preprocessing import LabelEncoder


app = Flask(__name__)

## load the model
model = pickle.load(open(r'C:\Carsales\Second-hand-car-price-prediction\dt_model.pkl','rb'))
label_encoder = LabelEncoder()


@app.route("/")
@app.route("/home")
def home_page():
    return render_template('home.html')

@app.route("/market")
def market_page():
    items = [
        {'id': 1, 'name': 'Phone', 'barcode': '893212299897', 'price': 500},
        {'id': 2, 'name': 'Laptop', 'barcode': '123985473165', 'price': 900},
        {'id': 3, 'name': 'Keyboard', 'barcode': '231985128446', 'price': 150}
    ]
    return render_template('market.html', items=items)


@app.route('/predict_api', methods = ['POST'])
def predict_api():

    data = request.json['data']
    print(data)
    data = pd.DataFrame(data, index=['1'])
    for i in data.columns:
        if data[i].dtype == object:
            data[i] = label_encoder.fit_transform(data[i])


    output = model.predict(data)
    print(output[0])
    return jsonify(output[0])


